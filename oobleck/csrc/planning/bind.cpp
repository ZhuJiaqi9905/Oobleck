#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <map>
#include <tuple>
#include "execution_result.h"
#include "pipeline_template.h"

namespace py = pybind11;
using namespace oobleck;

PYBIND11_MODULE(pipeline_template, m) {
  m.doc() = "Oobleck pipeline template module";

  py::class_<LayerExecutionResult>(m, "LayerExecutionResult")
      .def(py::init<const int, const double, const double,
                    const std::map<int, double>&, const std::map<int, double>&,
                    const std::tuple<int, int>&>(),
           py::arg("layer_index"), py::arg("forward"), py::arg("backward"),
           py::arg("allreduce_in_node"), py::arg("allreduce_across_nodes"),
           py::arg("mem_required"))
      .def_readonly("_index", &LayerExecutionResult::layer_index_)
      .def_readonly("_forward", &LayerExecutionResult::forward_)
      .def_readonly("_backward", &LayerExecutionResult::backward_)
      .def_readonly("_allreduce_in_node",
                    &LayerExecutionResult::allreduce_in_node_)
      .def_readonly("_allreduce_across_nodes",
                    &LayerExecutionResult::allreduce_across_nodes_)
      .def_readonly("_mem_required", &LayerExecutionResult::mem_required_);

  py::class_<LayerExecutionResults, std::shared_ptr<LayerExecutionResults>>(
      m, "LayerExecutionResults")
      .def(py::init<std::vector<LayerExecutionResult>&&>())
      .def("get", &LayerExecutionResults::get)
      .def("at", &LayerExecutionResults::at, py::arg("index"))
      .def_property_readonly("size", &LayerExecutionResults::size);

  py::class_<StageExecutionResult, std::shared_ptr<StageExecutionResult>>(
      m, "StageExecutionResult")
      .def(py::init<const std::shared_ptr<LayerExecutionResults>,
                    const std::tuple<int, int>&, const int>())
      .def_readonly("_num_gpus", &StageExecutionResult::num_gpus_)
      .def_readonly("_layer_indices", &StageExecutionResult::layer_indices_)
      .def_readonly("_forward", &StageExecutionResult::forward_)
      .def_readonly("_backward", &StageExecutionResult::backward_)
      .def_property_readonly("_num_layers", &StageExecutionResult::num_layers)
      .def_readonly("_mem_required", &StageExecutionResult::mem_required_);

  py::class_<PipelineTemplate>(m, "PipelineTemplate")
      .def(py::init<const std::vector<std::shared_ptr<StageExecutionResult>>&,
                    const double, const int, const int, const int>())
      .def("get_stages", &PipelineTemplate::get_stages)
      .def("get_rank_grid", &PipelineTemplate::get_rank_grid, py::arg("ranks"))
      .def_property_readonly("_iteration_time",
                             &PipelineTemplate::get_iteration_time)
      .def_property_readonly("_num_nodes", &PipelineTemplate::get_num_nodes)
      .def_property_readonly("_num_gpus_per_node",
                             &PipelineTemplate::get_num_gpus_per_node)
      .def("__repr__", [](const PipelineTemplate& pt) {
        return "<oobleck.PipelineTemplate." +
               std::to_string(pt.get_num_nodes()) + "nodes>";
      });

  py::class_<PipelineTemplateGenerator>(m, "PipelineTemplateGenerator")
      .def(py::init<>())
      .def("create_pipeline_templates",
           &PipelineTemplateGenerator::create_pipeline_templates)
      .def("create_pipeline_templates_serial",
           &PipelineTemplateGenerator::create_pipeline_templates_serial);
  m.def("get_profile_results", &get_profile_results,
        py::arg("model_tag"), py::arg("microbatch_size"), py::arg("world_size"), py::arg("num_workers_per_node"));
}