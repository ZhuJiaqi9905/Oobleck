import asyncio
import multiprocessing
import os
import socket
import sys
import netifaces
from dataclasses import dataclass
from multiprocessing import connection

import simple_parsing as sp
from deepspeed.utils.logging import LoggerFactory
from deepspeed.utils.logging import logging
from oobleck.elastic.worker import worker_main

import oobleck.elastic.message_util as message_util
from oobleck.csrc.planning.pipeline_template import get_profile_results
from oobleck.elastic.training_util import OobleckArguments
from oobleck.planning.profiler import profile, validate_model_args

logger = LoggerFactory.create_logger("oobleck_agent", logging.DEBUG)



def get_all_ip_addresses():
    ip_addresses = []
    interfaces = netifaces.interfaces()
    for interface in interfaces:
        addrs = netifaces.ifaddresses(interface)
        for family in (netifaces.AF_INET, netifaces.AF_INET6):
            if family in addrs:
                ip_addresses.extend(addr['addr'] for addr in addrs[family])
    return ip_addresses

@dataclass
class Worker:
    pipe: connection.Connection
    process: multiprocessing.Process


class OobleckAgent:
    """
    Oobleck agent process that runs on each agent node.
    It manages worker processes, where one worker is a rank in distributed training.

    An agent does:
    1. It registers itself to the master daemon when it starts.
    2. After registration, it periodically sends a liveness packet to the master,
       and wait for reconfiguration notification.
    3. Once reconfiguration is arrived, it sends a SIGUSR1 signal to all workers,
       letting them know that they need reconfiguration.
    4. An agent and workers have a dedicated mp.queue. After sending a signal,
       the agent queries a new distribution information from the master and forward it to workers.
    """

    def __init__(self, master_ip: str, master_port: int, job_id: int, agent_index: int):
        self._master_ip = master_ip
        self._master_port = master_port
        self._job_id = job_id
        self._agent_index = agent_index

        self._args: OobleckArguments | None = None
        self._conn: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None
        self._workers: list[Worker] = []
        self._response_callbacks: dict[message_util.RequestType, callable] = {}
        self._job_done: bool = False
        self._my_ip = ""

    async def run(self):
        await self._connect_to_master(self._master_ip, self._master_port)
        args = await self._register_agent(self._job_id)
        await self._launch_workers(args)

        self._args = args
        while not self._job_done:
            await self.on_receive_response()

    async def _connect_to_master(self, master_ip: str, master_port: int):
        # TODO: add timeout in connection
        self._conn = await asyncio.wait_for(
            asyncio.open_connection(master_ip, master_port),
            timeout=message_util.TIMEOUT,
        )

    async def _register_agent(self, job_id: int) -> OobleckArguments:
        await message_util.send_request_type(
            self._conn[1], message_util.RequestType.REGISTER_AGENT
        )
        await message_util.send(self._conn[1], job_id)
        result, req = await message_util.recv_response(self._conn[0])
        if (
            result is not message_util.Response.SUCCESS
            or req is not message_util.RequestType.REGISTER_AGENT
        ):
            raise ConnectionError("Failed to register agent")

        return await message_util.recv(self._conn[0])

    def _run_profiler(self, args: OobleckArguments):
        ctx = multiprocessing.get_context("spawn")
        profiler_processes: list[multiprocessing.Process] = []

        for index in range(args.dist.num_workers):
            # use CUDA_VISIBLE_DEVICES environment variable 
            os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
            my_ip = socket.gethostbyname(socket.gethostname())
            master_ip = args.dist.node_ips[0]
            master_port = 23456
            world_size = len(args.dist.node_ips) * args.dist.num_workers
            if args.dist.node_ips.count(my_ip) > 0:
                rank = args.dist.node_ips.index(my_ip) * args.dist.num_workers + index
            else:
                ip_address = get_all_ip_addresses()
                for ip in ip_address:
                    if args.dist.node_ips.count(ip) > 0:
                        my_ip = ip
                        rank = args.dist.node_ips.index(my_ip) * args.dist.num_workers + index
                        break 
            logger.info(f"my_ip: {my_ip}")
            # each worker run profile() in a new process
            process = ctx.Process(
                target=profile,
                args=(
                    args,
                    master_ip,
                    master_port,
                    args.dist.num_workers,
                    world_size,
                    rank,
                ),
            )
            process.start()
            profiler_processes.append(process)

        for process in profiler_processes:
            process.join()
        

    async def _launch_workers(self, args: OobleckArguments):
        # Test if profile data exists
        # 如果有profile数据就读出来
        try:
            get_profile_results(
                args.model.model_tag,
                args.job.microbatch_size,
                args.dist.world_size,
                args.dist.num_workers
            )

            if not validate_model_args(args):
                logger.warning(
                    "Model arguments are inconsistent: "
                    f"{args.model.model_args} != model_args.json."
                )
                raise RuntimeError("Model arguments are inconsistent")

            logger.info(f"Job arguments: {args}")
        except Exception:
            # Run profiler
            logger.warning(
                f"Profile data for model {args.model.model_name} not found. Launching profiler..."
            )
            self._run_profiler(args)

        dist_info = message_util.DistributionInfo(
            agent_ips=args.dist.node_ips,
            world_size=len(args.dist.node_ips) * args.dist.num_workers,
        )

        ctx = multiprocessing.get_context("spawn")
        #每个node可以启动多个agent。他们的agent_index是0, 1, 2...
        # 一个worker对应一个gpu

        gpu_indices = range(
            self._agent_index * args.dist.num_workers,
            (self._agent_index + 1) * args.dist.num_workers,
        )
        # If a worker has rank 0, it should forward its port to the master
        my_ip: str = socket.gethostbyname(socket.gethostname())
        if args.dist.node_ips.count(my_ip) <= 0:
            ip_address = get_all_ip_addresses()
            for ip in ip_address:
                if args.dist.node_ips.count(ip) > 0:
                    my_ip = ip
                    break 
        self._my_ip = my_ip
        logger.info(f"in agent. my_ip {my_ip}, node_ips {args.dist.node_ips}")
        for gpu_index in gpu_indices:
            logger.info(f"Launching worker {gpu_index}...")
            # TODO: add all arguments. Arguments should be passed from the master
            # via command line arguments.
            pipe, child_pipe = ctx.Pipe()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            # each worker run worker_main() in a new process
            process = ctx.Process(
                target=worker_main,
                args=(
                    gpu_index,
                    len(args.dist.node_ips),
                    args.dist.num_workers,
                    child_pipe,
                    my_ip,
                    args,
                ),
                daemon=True,
            )
            process.start()

            self._workers.append(Worker(pipe, process))
            pipe.send(dist_info)

            # TODO: detect worker failure and report it to the master.
            # For now only consider node failure, thus training will go stuck
            # if a worker fails.
        os.environ.pop("CUDA_VISIBLE_DEVICES")


        if my_ip == args.dist.node_ips[0]:
            await self.forward_worker_port(self._workers[0].pipe)
        

    async def forward_worker_port(self, pipe: connection.Connection):
        ''' 
        get the TCP Store port from worker and send it to master.
        '''
        _, w = self._conn
        port: int = pipe.recv()
        logger.info(f"Received worker port: {port} from worker. Forwarding it to master...")
        await message_util.send_request_type(
            w, message_util.RequestType.FORWARD_RANK0_PORT
        )
        await message_util.send(w, port, need_pickle=True, drain=True, close=False)

    async def on_receive_worker_port(self, port: int):
        '''
        The master send the TCP Store port to all agents.
        Each agent sned the port to its workers.
        '''
        logger.debug(f"agent recv TCP Store port {port} from master")
        r, w = self._conn

        for worker in self._workers:
            worker.pipe.send(port)

    async def send_request(
        self,
        request: message_util.RequestType,
        args: dict | None = None,
        callback: callable = None,
    ):
        if request in self._response_callbacks:
            logger.warning(
                f"Already pending request for the same request type {request}"
            )
            return

        if request is not message_util.RequestType.PING:
            self._response_callbacks[request] = callback
        await message_util.send_request_type(self._conn[1], request)

        if args is not None:
            await message_util.send(
                self._conn[1], args, need_pickle=True, drain=True, close=False
            )

    async def on_receive_reconfiguration(self, lost_node_ip: str):
        logger.debug(f"reconfiguration request received due to node failure: {lost_node_ip}")

        # This is for emulating a lost node by sending a command from the master.
        # Won't happen in normal case.
        if lost_node_ip == socket.gethostbyname(socket.gethostname()):
            logger.info("I'm the lost node. I'll terminate myself.")
            for worker in self._workers:
                worker.process.terminate()
            sys.exit(1)

        else:
            self._args.dist.node_ips.remove(lost_node_ip)
            # Send notification to workers
            for worker in self._workers:
                worker.pipe.send(lost_node_ip)
            if self._my_ip == self._args.dist.node_ips[0]:
                await self.forward_worker_port(self._workers[0].pipe)
            

    async def on_receive_response(self):
        r, w = self._conn
        loop = asyncio.get_running_loop()
        try:
            while not r.at_eof():
                result = await message_util.recv_response(r, timeout=None)
                logger.debug(f"Receiving: {result}")

                if result == (
                    message_util.Response.PONG,
                    message_util.RequestType.PING,
                ):
                    pass

                elif result == (
                    message_util.Response.RECONFIGURATION,
                    message_util.RequestType.UNDEFINED,
                ):
                    lost_node: str = await message_util.recv(
                        self._conn[0], need_pickle=True
                    )
                    loop.create_task(self.on_receive_reconfiguration(lost_node))

                elif result == (
                    message_util.Response.FORWARD_RANK0_PORT,
                    message_util.RequestType.UNDEFINED,
                ):
                    port: int = await message_util.recv(r, need_pickle=True)
                    loop.create_task(self.on_receive_worker_port(port))
                elif result[0] == message_util.Response.SUCCESS:
                    response, request = result
                    if request not in self._response_callbacks:
                        logger.warning(f"Unexpected response: {request}")
                        continue

                    callback = self._response_callbacks.pop(request)
                    await callback()
                else:
                    logger.warning(f"Unexpected response: {result}")
                    continue

        except asyncio.IncompleteReadError:
            logger.info("Connection closed by master")
            self._job_done = True
            return

    async def ping(self):
        loop = asyncio.get_running_loop()
        try:
            while loop.is_running():
                await asyncio.sleep(0.4)
                logger.debug("Sending ping")
                await self.send_request(message_util.RequestType.PING, None, None)
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    parser = sp.ArgumentParser()
    parser.add_argument("--master_ip", type=str)
    parser.add_argument("--master_port", type=int)
    parser.add_argument("--job_id", type=int)
    parser.add_argument("--agent_index", type=int)

    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_SOCKET_IFNAME"] = "eno1"
    # os.environ["GLOO_SOCKET_IFNAME"] = "eno1"
    # os.environ["NCCL_DESYNC_DEBUG"] = "1"
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    args = parser.parse_args()
    agent = OobleckAgent(
        args.master_ip, args.master_port, args.job_id, args.agent_index
    )
    asyncio.run(agent.run())
