from deepspeed.utils.logging import LoggerFactory
from deepspeed.utils.timer import SynchronizedWallClockTimer

logger = LoggerFactory.create_logger(__name__)
sync_timer = SynchronizedWallClockTimer()


def measure_time(timer_name: str):
    def inner(func: callable):
        def wrapper(s, *args, **kwargs):
            global sync_timer
            timer: SynchronizedWallClockTimer.Timer = sync_timer(timer_name)
            timer.start()
            # TODO: restore timer later.
            result = func(s, *args, **kwargs)
            timer.stop()
            return result

        return wrapper

    return inner




if __name__ == "__main__":
    
    @measure_time("test")
    def test_timer(s):
        sum = 0
        for i in range(0, 10000000):
            sum += i

    for i in range(0, 10):
        test_timer(1)
        sync_timer.log(["test"])
