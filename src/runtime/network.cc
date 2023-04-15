#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "flexflow/simulator.h"
namespace FlexFlow {
#define PRINT_EDGE(e, n)                                                       \
  do {                                                                         \
    std::cout << "(" << e / n << ", " << e % n << ")";                         \
  } while (0);

#define INSERT_OR_ADD(_map, _key, _val)                                        \
  do {                                                                         \
    if ((_map).find(_key) == (_map).end()) {                                   \
      (_map)[(_key)] = _val;                                                   \
    } else {                                                                   \
      (_map)[(_key)] += _val;                                                  \
    }                                                                          \
  } while (0);

static std::random_device rd;
static std::mt19937 gen = std::mt19937(rd());
static std::uniform_real_distribution<float> unif(0, 1);

// for summing connections...
template <typename T>
static std::vector<T> operator+(std::vector<T> const &a,
                                std::vector<T> const &b) {
  assert(a.size() == b.size());

  std::vector<T> result;
  result.reserve(a.size());

  std::transform(a.begin(),
                 a.end(),
                 b.begin(),
                 std::back_inserter(result),
                 std::plus<T>());
  return result;
}

WeightedShortestPathRoutingStrategy::WeightedShortestPathRoutingStrategy(
    ConnectionMatrix const &c,
    std::map<size_t, CommDevice *> const &devmap,
    int total_devs)
    : conn(c), devmap(devmap), total_devs(total_devs) {}

EcmpRoutes WeightedShortestPathRoutingStrategy::get_routes(int src_node,
                                                           int dst_node) {
  int key = src_node * total_devs + dst_node;

  if (conn[key] > 0) {
    return std::make_pair(std::vector<float>({1}),
                          std::vector<Route>({Route({devmap.at(key)})}));
  }

  // one-shortest path routing
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::priority_queue<std::pair<uint64_t, uint64_t>,
                      std::vector<std::pair<uint64_t, uint64_t>>,
                      std::greater<std::pair<uint64_t, uint64_t>>>
      pq;
  pq.push(std::make_pair(dist[src_node], src_node));
  dist[src_node] = 0;

  while (!pq.empty()) {
    int min_node = pq.top().second;
    pq.pop();
    visited[min_node] = true;

    if (min_node == dst_node) {
      break;
    }

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1; // numeric_limits<uint64_t>::max() /
                                           // get_bandwidth_bps(min_node, i);
      if (new_dist < dist[i] || (new_dist == dist[i] && unif(gen) < 0.5)) {
        dist[i] = new_dist;
        prev[i] = min_node;
        pq.push(std::make_pair(new_dist, i));
      }
    }
  }

  Route result = Route();
  int curr = dst_node;
  while (prev[curr] != -1) {
    result.insert(result.begin(), devmap.at(prev[curr] * total_devs + curr));
    curr = prev[curr];
  }
  assert(result.size() || src_node == dst_node);
  return std::make_pair(std::vector<float>{1}, std::vector<Route>{result});
}

void WeightedShortestPathRoutingStrategy::hop_count(int src_node,
                                                    int dst_node,
                                                    int &hop,
                                                    int &narrowest) {
  int key = src_node * total_devs + dst_node;

  if (conn[key] > 0) {
    hop = 0;
    narrowest = conn[key];
    return;
  }
  // one-shortest path routing
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::priority_queue<std::pair<uint64_t, uint64_t>,
                      std::vector<std::pair<uint64_t, uint64_t>>,
                      std::greater<std::pair<uint64_t, uint64_t>>>
      pq;
  pq.push(std::make_pair(dist[src_node], src_node));
  dist[src_node] = 0;
  while (!pq.empty()) {
    int min_node = pq.top().second;
    pq.pop();
    visited[min_node] = true;
    if (min_node == dst_node) {
      break;
    }
    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1; // numeric_limits<uint64_t>::max() /
                                           // get_bandwidth_bps(min_node, i);
      if (new_dist < dist[i]) {
        dist[i] = new_dist;
        prev[i] = min_node;
        pq.push(std::make_pair(new_dist, i));
      }
    }
  }
  hop = 0;
  narrowest = std::numeric_limits<int>::max();
  int curr = dst_node;
  while (prev[curr] != -1) {
    if (narrowest > conn[prev[curr] * total_devs + curr]) {
      narrowest = conn[prev[curr] * total_devs + curr];
    }
    hop++;
    curr = prev[curr];
  }
  assert(hop > 0 || src_node == dst_node);
}

std::vector<EcmpRoutes>
    WeightedShortestPathRoutingStrategy::get_routes_from_src(int src_node) {
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::priority_queue<std::pair<uint64_t, uint64_t>,
                      std::vector<std::pair<uint64_t, uint64_t>>,
                      std::greater<std::pair<uint64_t, uint64_t>>>
      pq;
  pq.push(std::make_pair(dist[src_node], src_node));
  dist[src_node] = 0;
  while (!pq.empty()) {
    int min_node = pq.top().second;
    pq.pop();
    visited[min_node] = true;

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1; // numeric_limits<uint64_t>::max() /
                                           // get_bandwidth_bps(min_node, i);
      if (new_dist < dist[i]) {
        dist[i] = new_dist;
        prev[i] = min_node;
        pq.push(std::make_pair(new_dist, i));
      }
    }
  }
  std::vector<EcmpRoutes> final_result;
  for (int i = 0; i < total_devs; i++) {
    if (i == src_node) {
      final_result.emplace_back(
          std::make_pair(std::vector<float>{}, std::vector<Route>{}));
      continue;
    }
    Route result = Route();
    int curr = i;
    while (prev[curr] != -1) {
      result.insert(result.begin(), devmap.at(prev[curr] * total_devs + curr));
      curr = prev[curr];
    }
    assert(result.size() > 0);
    final_result.emplace_back(
        std::make_pair(std::vector<float>{1}, std::vector<Route>{result}));
  }
  return final_result;
}

std::vector<std::pair<int, int>>
    WeightedShortestPathRoutingStrategy::hop_count(int src_node) {
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::priority_queue<std::pair<uint64_t, uint64_t>,
                      std::vector<std::pair<uint64_t, uint64_t>>,
                      std::greater<std::pair<uint64_t, uint64_t>>>
      pq;
  pq.push(std::make_pair(dist[src_node], src_node));
  dist[src_node] = 0;
  while (!pq.empty()) {
    int min_node = pq.top().second;
    pq.pop();
    visited[min_node] = true;

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1; // numeric_limits<uint64_t>::max() /
                                           // get_bandwidth_bps(min_node, i);
      if (new_dist < dist[i]) {
        dist[i] = new_dist;
        prev[i] = min_node;
        pq.push(std::make_pair(new_dist, i));
      }
    }
  }

  std::vector<std::pair<int, int>> result;
  for (int i = 0; i < total_devs; i++) {
    if (i == src_node) {
      result.emplace_back(std::make_pair(-1, 0));
      continue;
    }
    int hop = -1;
    int narrowest = 0;
    int curr = i;
    while (prev[curr] != -1) {
      if (!narrowest || (narrowest > conn[prev[curr] * total_devs + curr])) {
        narrowest = conn[prev[curr] * total_devs + curr];
      }
      hop++;
      curr = prev[curr];
    }
    result.emplace_back(std::make_pair(hop, narrowest));
  }
  return result;
}

ShortestPathNetworkRoutingStrategy::ShortestPathNetworkRoutingStrategy(
    ConnectionMatrix const &c,
    std::map<size_t, CommDevice *> const &devmap,
    int total_devs)
    : conn(c), devmap(devmap), total_devs(total_devs) {}

EcmpRoutes ShortestPathNetworkRoutingStrategy::get_routes(int src_node,
                                                          int dst_node) {
  int key = src_node * total_devs + dst_node;
  // std::cerr << "routing " << src_node << ", " << dst_node << std::endl;

  if (conn[key] > 0) {
    return std::make_pair(std::vector<float>({1}),
                          std::vector<Route>({Route({devmap.at(key)})}));
  }

  // one-shortest path routing
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::queue<uint64_t> q;
  q.push(src_node);
  dist[src_node] = 0;

  // BFS
  while (!q.empty()) {
    int min_node = q.front();
    q.pop();
    visited[min_node] = true;

    if (min_node == dst_node) {
      break;
    }

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1;
      if (new_dist < dist[i] || (new_dist == dist[i] && unif(gen) < 0.5)) {
        dist[i] = new_dist;
        prev[i] = min_node;
        q.push(i);
      }
    }
  }

  Route result = Route();
  int curr = dst_node;
  while (prev[curr] != -1) {
    result.insert(result.begin(), devmap.at(prev[curr] * total_devs + curr));
    curr = prev[curr];
  }
  assert(result.size() || src_node == dst_node);
  return std::make_pair(std::vector<float>{1}, std::vector<Route>{result});
}

std::vector<EcmpRoutes>
    ShortestPathNetworkRoutingStrategy::get_routes_from_src(int src_node) {
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::queue<uint64_t> q;
  q.push(src_node);
  dist[src_node] = 0;

  // BFS
  while (!q.empty()) {
    int min_node = q.front();
    q.pop();
    visited[min_node] = true;

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1;
      if (new_dist < dist[i] || (new_dist == dist[i] && unif(gen) < 0.5)) {
        dist[i] = new_dist;
        prev[i] = min_node;
        q.push(i);
      }
    }
  }

  std::vector<EcmpRoutes> final_result;
  for (int i = 0; i < total_devs; i++) {
    if (i == src_node) {
      final_result.emplace_back(
          std::make_pair(std::vector<float>{}, std::vector<Route>{}));
      continue;
    }
    Route result = Route();
    int curr = i;
    while (prev[curr] != -1) {
      result.insert(result.begin(), devmap.at(prev[curr] * total_devs + curr));
      curr = prev[curr];
    }
    // assert(result.size() > 0);
    final_result.emplace_back(
        std::make_pair(std::vector<float>{1}, std::vector<Route>{result}));
  }
  return final_result;
}

void ShortestPathNetworkRoutingStrategy::hop_count(int src_node,
                                                   int dst_node,
                                                   int &hop,
                                                   int &narrowest) {
  int key = src_node * total_devs + dst_node;

  if (conn[key] > 0) {
    hop = 0;
    narrowest = conn[key];
    return;
  }
  // one-shortest path routing
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::queue<uint64_t> q;
  q.push(src_node);
  dist[src_node] = 0;

  // BFS
  while (!q.empty()) {
    int min_node = q.front();
    q.pop();
    visited[min_node] = true;

    if (min_node == dst_node) {
      break;
    }

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1;
      if (new_dist < dist[i] || (new_dist == dist[i] && unif(gen) < 0.5)) {
        dist[i] = new_dist;
        prev[i] = min_node;
        q.push(i);
      }
    }
  }
  hop = 0;
  narrowest = std::numeric_limits<int>::max();
  int curr = dst_node;
  while (prev[curr] != -1) {
    if (narrowest > conn[prev[curr] * total_devs + curr]) {
      narrowest = conn[prev[curr] * total_devs + curr];
    }
    hop++;
    curr = prev[curr];
  }
  assert(hop > 0 || src_node == dst_node);
}

std::vector<std::pair<int, int>>
    ShortestPathNetworkRoutingStrategy::hop_count(int src_node) {
  std::vector<uint64_t> dist(total_devs, std::numeric_limits<uint64_t>::max());
  std::vector<int> prev(total_devs, -1);
  std::vector<bool> visited(total_devs, false);

  std::queue<uint64_t> q;
  q.push(src_node);
  dist[src_node] = 0;

  // BFS
  while (!q.empty()) {
    int min_node = q.front();
    q.pop();
    visited[min_node] = true;

    for (int i = 0; i < total_devs; i++) {
      if (visited[i] || conn[min_node * total_devs + i] == 0) {
        continue;
      }
      float new_dist = dist[min_node] + 1;
      if (new_dist < dist[i] || (new_dist == dist[i] && unif(gen) < 0.5)) {
        dist[i] = new_dist;
        prev[i] = min_node;
        q.push(i);
      }
    }
  }

  std::vector<std::pair<int, int>> result;
  for (int i = 0; i < total_devs; i++) {
    if (i == src_node) {
      result.emplace_back(std::make_pair(-1, 0));
      continue;
    }
    int hop = -1;
    int narrowest = 0;
    int curr = i;
    while (prev[curr] != -1) {
      if (!narrowest || (narrowest > conn[prev[curr] * total_devs + curr])) {
        narrowest = conn[prev[curr] * total_devs + curr];
      }
      hop++;
      curr = prev[curr];
    }
    result.emplace_back(std::make_pair(hop, narrowest));
  }
  return result;
}

WeightedMultiplePathRoutingStrategy::WeightedMultiplePathRoutingStrategy(ConnectionMatrix const &c,
    std::map<size_t, CommDevice *> const &devmap,
    int total_devs,
    int total_nodes) : conn(c),
                       state(c),                       // shape 是 1, total_devs * total_devs的
                       devmap(devmap),                 // shape 是 1, total_devs * total_devs的
                       total_devs(total_devs),
                       total_nodes(total_nodes) {
  routes_memory.clear();
  devmap_id.clear();
  for(auto k : devmap) {
    devmap_id.insert(std::pair<CommDevice *, size_t>(k.second, k.first));
  }
  get_routes_all();
}

EcmpRoutes WeightedMultiplePathRoutingStrategy::get_routes(int src_node,
                                                           int dst_node) {
  assert(routes_memory.count(std::pair<int,int>(src_node, dst_node)) > 0);
  //EcmpRoutes ret = routes_memory.at(std::make_pair<int,int>(src_node, dst_node));
  //for(int i = 0; i < ret.first.size(); i++) {
  //  if(!check_routes_capacity(ret.second[i])) {
  //    ret.first[i] = -1;
  //  }
  //}
  //return ret;
  return routes_memory.at(std::pair<int,int>(src_node, dst_node));
}

void WeightedMultiplePathRoutingStrategy::hop_count(int src_node,
                                                    int dst_node,
                                                    int &hop,
                                                    int &narrowest) {
  EcmpRoutes candidate_path = get_routes(src_node, dst_node);
  int num_candidates = candidate_path.first.size();
  for(int i = 0; i < num_candidates; i++) {
    bool available_path = true;
    for(int j = 0; j < candidate_path.second[i].size(); j++) {
      if(state[devmap_id[candidate_path.second[i][j]]] == 0) {
        available_path = false;
        break;
      }
    }
    if(available_path) {
      hop = hop > candidate_path.second[i].size() ? candidate_path.second[i].size() : hop;
      narrowest = 1;
    }
  }
  return;
}

std::vector<EcmpRoutes> WeightedMultiplePathRoutingStrategy::get_routes_from_src(int src_node) {
  std::vector<EcmpRoutes> final_result;
  for(int i = 0; i < total_devs; i++) {
    if(i == src_node) {
      final_result.emplace_back(std::make_pair(std::vector<float>{}, std::vector<Route>{}));
    } else {
      final_result.emplace_back(get_routes(src_node, i));
    }
  }
  return final_result;
}

void WeightedMultiplePathRoutingStrategy::clear() {
  state.clear();
  state.assign(conn.begin(), conn.end());
}


//bool WeightedMultiplePathRoutingStrategy::check_routes_capacity(Route &route) {
//  bool result = true;
//  for(auto dev: route) {
//    if(state[devmap_id[dev]] == 0) {
//      result = false;
//      break;
//    }
//  }
//  return result;
//}

void WeightedMultiplePathRoutingStrategy::get_routes_all() {
  int tolerance = 2; // 最短路和最长路的跳数差
  for(int src_node = 0; src_node < total_devs; src_node ++) {
    for(int dst_node = 0; dst_node < total_devs; dst_node ++) {
      if(src_node == dst_node) {
        routes_memory.insert(std::make_pair<std::pair<int,int>, EcmpRoutes>(
          std::pair<int,int>(src_node, dst_node),
          std::make_pair(std::vector<float>{}, std::vector<Route>{})));
        continue;
      }
      // 开始搜索BFS
      // 找多路的算法这里可能还需要想一想, 复杂度太高了
      EcmpRoutes src_dst_emcp;
      std::deque<std::pair<int, Route>> dq;
      bool has_found = false;
      int min_length = std::numeric_limits<int>::max();


      dq.emplace_back(std::pair<int, Route>(src_node, std::vector<CommDevice *>() ));

      while (!dq.empty()) {
        std::pair<int, Route> curr_node = dq.front();
        dq.pop_front();
        int curr_node_idx = curr_node.first;

        // check in the tolerated degree
        if (curr_node.second.size()-tolerance >= min_length-1) {
          continue;
        }
        // check is dst_node
        if(conn[total_devs * curr_node_idx + dst_node] > 0) {
          std::pair<int, Route> explo = std::pair<int, Route>(dst_node, curr_node.second);
          explo.second.emplace_back(devmap[total_devs * curr_node_idx + dst_node]);

          src_dst_emcp.first.emplace_back(1.0f);
          src_dst_emcp.second.emplace_back(explo.second);

          has_found = true;
          min_length = min_length < explo.second.size() ? min_length : explo.second.size();

          continue;
        }
        // find neighbors
        for(int neighbor_idx = total_nodes; neighbor_idx < total_devs; neighbor_idx ++) {
          if(conn[total_devs * curr_node_idx + neighbor_idx] > 0) {
            std::pair<int, Route> explo = std::pair<int, Route>(neighbor_idx, curr_node.second);
            bool no_repeated = true;

            // 去除重复经过的节点
            for(auto prev_dev: explo.second) {
              int prev_src = devmap_id[prev_dev] / total_devs;
              int prev_dst = devmap_id[prev_dev] % total_devs;
              if(prev_src == curr_node_idx && prev_dst == neighbor_idx) {
                no_repeated = false;
                break;
              }
              if(prev_src == neighbor_idx && prev_dst == curr_node_idx) {
                no_repeated = false;
                break;
              }
            }
            if(no_repeated) {
              explo.second.emplace_back(devmap[total_devs * curr_node_idx + neighbor_idx]);
              dq.emplace_back(explo);
            }
          }
        }
      }
      // standarize
      for(int i = 0; i < src_dst_emcp.first.size(); i++) {
        src_dst_emcp.first[i] = 1.0/src_dst_emcp.first.size();
      }
      routes_memory.insert(std::pair<std::pair<int,int>, EcmpRoutes>(
          std::pair<int,int>(src_node, dst_node),
          src_dst_emcp));
    }
  }
}

FlatDegConstraintNetworkTopologyGenerator::
    FlatDegConstraintNetworkTopologyGenerator(int num_nodes, int degree)
    : num_nodes(num_nodes), degree(degree) {}

ConnectionMatrix
    FlatDegConstraintNetworkTopologyGenerator::generate_topology() const {
  ConnectionMatrix conn = std::vector<int>(num_nodes * num_nodes, 0);

  int allocated = 0;
  int curr_node = 0;
  std::unordered_set<int> visited_node;
  visited_node.insert(0);

  std::uniform_int_distribution<> distrib(0, num_nodes - 1);

  while ((long)visited_node.size() != num_nodes) {
    distrib(gen);
    int next_step = distrib(gen);
    if (next_step == curr_node) {
      continue;
    }
    if (visited_node.find(next_step) == visited_node.end()) {
      if (conn[get_id(curr_node, next_step)] == degree) {
        continue;
      }
      conn[get_id(curr_node, next_step)]++;
      conn[get_id(next_step, curr_node)]++;
      visited_node.insert(next_step);
      curr_node = next_step;
      allocated += 2;
    }
  }

  assert(allocated == (num_nodes - 1) * 2);

  std::vector<std::pair<int, int>> node_with_avail_if;
  for (int i = 0; i < num_nodes; i++) {
    int if_inuse = get_if_in_use(i, conn);
    if (if_inuse < degree) {
      node_with_avail_if.emplace_back(i, degree - if_inuse);
    }
  }

  distrib = std::uniform_int_distribution<>(0, node_with_avail_if.size() - 1);
  int a = 0, b = 0;

  while (node_with_avail_if.size() > 1) {
    a = distrib(gen);
    while ((b = distrib(gen)) == a) {
      ;
    }

    assert(
        conn[get_id(node_with_avail_if[a].first, node_with_avail_if[b].first)] <
        degree);
    conn[get_id(node_with_avail_if[a].first, node_with_avail_if[b].first)]++;
    conn[get_id(node_with_avail_if[b].first, node_with_avail_if[a].first)]++;
    allocated += 2;

    bool changed = false;
    if (--node_with_avail_if[a].second == 0) {
      if (a < b) {
        b--;
      }
      node_with_avail_if.erase(node_with_avail_if.begin() + a);
      changed = true;
    }
    if (--node_with_avail_if[b].second == 0) {
      node_with_avail_if.erase(node_with_avail_if.begin() + b);
      changed = true;
    }
    if (changed) {
      distrib =
          std::uniform_int_distribution<>(0, node_with_avail_if.size() - 1);
    }
  }

#ifdef DEBUG_PRINT
  std::cout << "Topology generated: " << std::endl;
  NetworkTopologyGenerator::print_conn_matrix(conn, num_nodes, 0);
#endif
  return conn;
}

int FlatDegConstraintNetworkTopologyGenerator::get_id(int i, int j) const {
  return i * num_nodes + j;
}

int FlatDegConstraintNetworkTopologyGenerator::get_if_in_use(
    int node, ConnectionMatrix const &conn) const {
  int result = 0;
  for (int i = 0; i < num_nodes; i++) {
    result += conn[get_id(node, i)];
  }
  return result;
}

BigSwitchNetworkTopologyGenerator::BigSwitchNetworkTopologyGenerator(
    int num_nodes)
    : num_nodes(num_nodes) {}

ConnectionMatrix BigSwitchNetworkTopologyGenerator::generate_topology() const {
  ConnectionMatrix conn =
      std::vector<int>((num_nodes + 1) * (num_nodes + 1), 0);
  for (int i = 0; i < num_nodes; i++) {
    conn[i * (num_nodes + 1) + num_nodes] = 1;
    conn[num_nodes * (num_nodes + 1) + i] = 1;
  }
  return conn;
}

ConnectionMatrix FatTreeTopologyGenerator::generate_topology() const {
  // something needs to be rethought is whether the connections in the core layer is necessary

  int num_nodes = actual_server_ids.size();
  int num_nodes_in_network = num_nodes_per_pod * num_pods;
  int num_lower_sw = num_pods * num_sw_per_pod / 2;
  int num_upper_sw = num_lower_sw;
  int num_lower_sw_per_pod = num_sw_per_pod / 2;
  int num_upper_sw_per_pod = num_sw_per_pod / 2;
  int num_nodes_per_sw = num_nodes_in_network/num_lower_sw;
  int num_devices = num_nodes + num_pods * num_sw_per_pod + num_pods;
  int num_core = num_lower_sw * num_upper_sw;

  ConnectionMatrix conn = 
      std::vector<int>(num_devices * num_devices, 0);
  
  // connect server to lower switches
  for(int i = 0; i < actual_server_ids.size(); i++) {
    int neighbor_sw_id = num_nodes + actual_server_ids[i] / num_nodes_per_sw;
    conn[num_devices * i+ neighbor_sw_id] = degree_node_to_sw;
    conn[num_devices * neighbor_sw_id + i] = degree_node_to_sw;
  }

  // connect lower switches to higher switches
  for(int lower_sw_id = num_nodes;
      lower_sw_id < num_nodes + num_lower_sw;
      lower_sw_id ++) {
    for(int j = 0; j < num_upper_sw_per_pod; j++) {
      int upper_neighbor_sw_id = num_nodes + num_lower_sw + (lower_sw_id-num_nodes) / num_lower_sw_per_pod * num_lower_sw_per_pod + j;
      conn[num_devices * upper_neighbor_sw_id + lower_sw_id] = degree_sw_intra_pod;
      conn[num_devices * lower_sw_id + upper_neighbor_sw_id] = degree_sw_intra_pod;
    }
  }

  // connect upper switches to core switches
  for(int upper_sw_id = num_nodes + num_lower_sw;
      upper_sw_id < num_nodes + num_lower_sw + num_upper_sw;
      upper_sw_id ++) {
    for(int j = 0; j < num_upper_sw_per_pod; j++) {
      int core_neighbor_sw_id = num_nodes+num_lower_sw+num_upper_sw + (upper_sw_id - num_nodes - num_lower_sw) % num_upper_sw_per_pod * num_upper_sw_per_pod + j;
      conn[num_devices * upper_sw_id + core_neighbor_sw_id] = degree_sw_inter_pod;
      conn[num_devices * core_neighbor_sw_id + upper_sw_id] = degree_sw_inter_pod;
    }
  }

  return conn;
}

CustomTopologyGenerator::CustomTopologyGenerator(std::string file) {
  // file 的格式
  // 开头三行是num_node 和 num switches 和 gpu_per_node

  // 接下来是邻接矩阵开头用 > 表示开始
  printf("filename %s\n", file.c_str());
  std::ifstream machine_config(file);
  std::string line;
  while (std::getline(machine_config, line)) {
    if (line[0] != '#') {
      std::istringstream iss(line);
      std::vector<std::string> words{std::istream_iterator<std::string>{iss},
                                     std::istream_iterator<std::string>{}};
      if(words[0] == "num_nodes") {
        num_nodes = std::stoi(words[2]);
      } else if(words[0] == "num_switches") {
        num_switches = std::stoi(words[2]);
      } else if(words[0] == "gpu_per_node") {
        gpu_per_node = std::stoi(words[2]);
      } else if(words[0] == ">") {
        for(int i = 0; i < num_switches+num_nodes; i++) {
          conn.emplace_back(std::stoi(words[i+1]));
        }
      } else {
        assert(false);
      }
    }
  }
}

ConnectionMatrix CustomTopologyGenerator::generate_topology() const {
  return conn;
}

}; // namespace FlexFlow