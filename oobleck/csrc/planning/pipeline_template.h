#ifndef _OOBLECK_PLANNING_PIPELINE_TEMPLATE_H_
#define _OOBLECK_PLANNING_PIPELINE_TEMPLATE_H_

#include <oneapi/tbb/concurrent_hash_map.h>
#include <pybind11/pybind11.h>
#include <atomic>
#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>
#include "execution_result.h"

namespace oobleck {

class PipelineTemplate {
 public:
  PipelineTemplate(const std::vector<std::shared_ptr<StageExecutionResult>>&
                       stage_execution_results,
                   const double iteration_time,
                   const int num_layers,
                   const int num_nodes,
                   const int num_gpus_per_node)
      : stage_execution_results_(stage_execution_results),
        iteration_time_(iteration_time),
        num_nodes_(num_nodes),
        num_gpus_per_node_(num_gpus_per_node) {
    // Run divide and conquer to create a vector of StageExecutionResult
    // Perform assertion
    // 1. num_nodes * num_gpus_per_node == all # GPUs used by stage results
    // 2. stages cover all layers
    int num_gpus_used = 0;
    for (auto& stage : stage_execution_results_) {
      std::cout << stage->to_string() << std::endl;
      num_gpus_used += stage->num_gpus_;
    }
    assert(num_gpus_used == num_nodes * num_gpus_per_node);

    int stage_num_layers = 0;
    for (auto& stage : stage_execution_results_) {
      stage_num_layers += stage->num_layers();
    }
    assert(stage_num_layers == num_layers);
  }

  const double get_iteration_time() const { return iteration_time_; }
  const std::vector<std::shared_ptr<StageExecutionResult>>& get_stages() const {
    return stage_execution_results_;
  }
  int get_num_nodes() const { return num_nodes_; }
  int get_num_gpus_per_node() const { return num_gpus_per_node_; }

  std::map<int, std::vector<int>> get_rank_grid(std::vector<int> ranks) {
    // Return a map of layer index to a list of ranks
    std::map<int, std::vector<int>> rank_grid;
    for (auto& stage : stage_execution_results_) {
      std::vector<int> stage_ranks(ranks.begin(),
                                   ranks.begin() + stage->num_gpus_);
      ranks.erase(ranks.begin(), ranks.begin() + stage->num_gpus_);

      std::vector<int> layer_ranks(num_gpus_per_node_);
      auto it = layer_ranks.begin();

      // If length of `stage_ranks` is less than num_gpus_per_node,
      // adjust it so that it has the same length as num_gpus_per_node
      const int repeat_count = num_gpus_per_node_ / stage->num_gpus_;
      for (const int rank : stage_ranks) {
        std::fill_n(it, repeat_count, rank);
        std::advance(it, repeat_count);
      }

      // push per-layer ranks to the result
      for (const int layer_index : stage->layer_indices_) {
        rank_grid[layer_index] = layer_ranks;
      }
    }

    assert(ranks.size() == 0);
    return std::move(rank_grid);
  }

 private:
  std::vector<std::shared_ptr<StageExecutionResult>> stage_execution_results_;
  const double iteration_time_;
  const int num_nodes_;
  const int num_gpus_per_node_;
};

std::shared_ptr<LayerExecutionResults> get_profile_results(
    const std::string& model_tag,
    const int microbatch_size,
    const int world_size,
    const int num_workers_per_node);

class PipelineTemplateGenerator {
 public:
  CacheMap dc_cache_;
  cppcoro::static_thread_pool thread_pool_;

  std::vector<PipelineTemplate> create_pipeline_templates(
      std::shared_ptr<LayerExecutionResults> layer_execution_results,
      const std::tuple<int, int>& num_nodes,
      const int num_gpus_per_node);

  std::vector<PipelineTemplate> create_pipeline_templates_serial(
    std::shared_ptr<LayerExecutionResults> layer_execution_results,
    const std::tuple<int, int>& num_nodes,
    const int num_gpus_per_node);

 private:
  cppcoro::task<std::shared_ptr<DCExecutionResult>> divide_and_conquer(
      std::shared_ptr<LayerExecutionResults> layer_execution_results,
      const std::tuple<int, int> layer_indices,
      const int num_stages,
      const int num_nodes,
      const int num_gpus_per_node);
    
  std::shared_ptr<DCExecutionResult>
    divide_and_conquer_serial(
    std::shared_ptr<LayerExecutionResults> layer_execution_results,
    const std::tuple<int, int> layer_indices,
    const int num_stages,
    const int num_nodes,
    const int num_gpus_per_node);


  std::atomic<unsigned long> cache_hit_{0};
  std::atomic<unsigned long> cache_miss_{0};
};

}  // namespace oobleck

#endif