#coding: utf8
from copy import deepcopy
from collections import defaultdict
import re
from recursive.cache import Cache
import os
from loguru import logger
import json
from datetime import datetime
import time
import hashlib
from functools import lru_cache


# article = ""
caches = {
    "search": None,
    # "llm": None,
    "web_page": None
}


class Memory:
    def __init__(self, root_node, format, config):
        self.root_node = root_node
        self.init()
        self.format = format
        self.config = config
        self.article = ""
        self.all_search_results = []
        self.global_start_index = 1
        assert self.format in ("xml", "nl")

        self.folder = ""
        
        # 内存缓存字典
        self._node_info_cache = {}
        self._collect_cache = {}
        self._cache_max_size = 1000  # 缓存最大条目数
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        
    def _generate_cache_key(self, prefix, *args):
        """
        生成缓存键，基于参数内容的哈希值
        """
        content = str(args)
        cache_hash = hashlib.blake2b(content.encode('utf-8'), digest_size=16).hexdigest()
        return f"{prefix}_{cache_hash}"
    
    def _get_from_cache(self, cache_key):
        """
        从缓存获取数据
        """
        if cache_key in self._node_info_cache:
            self._cache_hit_count += 1
            return self._node_info_cache[cache_key]
        self._cache_miss_count += 1
        return None
    
    def _set_to_cache(self, cache_key, value):
        """
        设置缓存数据，实现LRU策略
        """
        if len(self._node_info_cache) >= self._cache_max_size:
            # 简单的LRU实现：删除最早的25%条目
            keys_to_remove = list(self._node_info_cache.keys())[:self._cache_max_size // 4]
            for key in keys_to_remove:
                del self._node_info_cache[key]
        
        self._node_info_cache[cache_key] = deepcopy(value)
    
    def _clear_cache(self):
        """
        清空缓存
        """
        self._node_info_cache.clear()
        self._collect_cache.clear()
    
    def get_cache_stats(self):
        """
        获取缓存统计信息
        """
        total_requests = self._cache_hit_count + self._cache_miss_count
        hit_rate = (self._cache_hit_count / total_requests * 100) if total_requests > 0 else 0
        return {
            "cache_size": len(self._node_info_cache),
            "hit_count": self._cache_hit_count,
            "miss_count": self._cache_miss_count,
            "hit_rate": f"{hit_rate:.2f}%"
        }
        
        # 内存缓存字典
        self._node_info_cache = {}
        self._collect_cache = {}
        self._cache_max_size = 1000  # 缓存最大条目数
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        


    def add_article(self, content):
        if not content.strip():
            return

        self.article += "\n" + content

    def get_article(self):
        return self.article

    def get_article_latest(self, length=6000):
        if len(self.article) <= length:
            return self.article

        # 寻找合适的分割点（在句号、换行符等处分割，避免截断句子）
        target_start = len(self.article) - length
        
        # 从目标位置向前搜索分割点
        for i in range(target_start, len(self.article)):
            if self.article[i] in '。！？\n':
                return self.article[i + 1:]
        
        # 如果没找到合适的分割点，直接返回后length个字符
        return self.article[-length:]

    def add_search_result(self, page):
        # page["global_index"] = len(self.all_search_results) + 1
        self.all_search_results.append(page)
        self.global_start_index += 1
        return page
  

    def init(self):
        self.info_nodes = {
            self.root_node.hashkey: InfoNode(
                self.root_node.hashkey, self.root_node.nid, None, [], 0, self.root_node.task_info)
        } # key: hashkey, value: infonode

    def save(self, folder):
        import json
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        
        article_file_path = "{}/article.txt".format(folder)
        with open(article_file_path, 'w', encoding='utf-8') as file:
            file.write(self.article)

        with open("{}/memory.jsonl".format(folder), "w", encoding='utf-8') as f:
            f.write(json.dumps({
                "all_search_results": self.all_search_results
            }, ensure_ascii=False))

    def load(self, folder):
        self.folder = folder

        article_file_path = "{}/article.txt".format(folder)
        if os.path.exists(article_file_path):
            with open(article_file_path, 'r', encoding='utf-8') as file:
                self.article = file.read()

        memory_file = "{}/memory.jsonl".format(folder)
        if os.path.exists(memory_file):
            with open(memory_file, "r", encoding='utf-8') as f:
                data = json.loads(f.read())
                self.all_search_results = data.get("all_search_results", [])
                self.global_start_index = len(self.all_search_results) + 1

    def database_set(self, key, value):
        # if self.multiprocess_manager is not None:
        self.database[key] = value
       
    def get_database_key(self, key):
        pass
            
    def load_database(self):
        pass

    def collect_infos(self, node_list):
        """
        收集节点信息，添加缓存支持
        """
        # 生成基于节点列表的缓存键
        node_keys = [f"{node.hashkey}_{node.nid}" for node in node_list]
        cache_key = self._generate_cache_key("collect_infos", *sorted(node_keys))
        
        # 检查缓存
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for collect_infos: {len(node_list)} nodes")
            self.info_nodes = cached_result
            return
            
        logger.info(f"Cache miss for collect_infos: {len(node_list)} nodes, computing...")
        
        # 原有逻辑
        self.init()
        def inner(node):
            if node.hashkey in self.info_nodes:
                return self.info_nodes[node.hashkey]
            # If it doesn't exist, create a new infoNode
            # First create its external nodes and dependent nodes
            outer_info_node = inner(node.node_graph_info["outer_node"])
            parent_info_nodes = [inner(parent) for parent in node.node_graph_info["parent_nodes"]]
        
            info = deepcopy(node.task_info)
            info["final_result"] = node.get_node_final_result()
            
            info_node = InfoNode(node.hashkey, node.nid, outer_info_node, parent_info_nodes, 
                                 node.node_graph_info["layer"], info)
            self.info_nodes[node.hashkey] = info_node
            return info_node
        for node in node_list:
            inner(node)
            
        # 缓存结果
        self._set_to_cache(cache_key, deepcopy(self.info_nodes))
            
    def update_infos(self, node_list):
        """
        node_list consists of nodes that need information updates
        """
        self.collect_infos(node_list)

    def _process_node_info(self, cur):
        if self.format == "xml":
            content = """
    <任务 id={}>
    <依赖任务>
    {}
    </依赖任务>
    <任务目标>
    {}
    </任务目标>
    <任务结果>
    {}
    </任务结果>
    </任务>
    """.format(cur.nid, 
            ",".join(str(par.nid) for par in cur.parent_nodes) if len(cur.parent_nodes) > 0 else "无", 
            cur.info["goal"], cur.info["final_result"]["result"]).strip()
        elif self.format == "nl":
            content = "{}. {}: \n{}\n".format(cur.nid, cur.info["goal"], cur.info["final_result"]["result"])
        return content 
        
    def get_json_node_info(self, graph_node):
        if graph_node.task_type_tag == "COMPOSITION":
            return None
        info_node = self.info_nodes[graph_node.hashkey]
        represent = {
            "id": graph_node.nid,
            "task_type": graph_node.task_info["task_type"],
            "goal": graph_node.task_info["goal"],
            "dependency": [n.nid for n in graph_node.node_graph_info["parent_nodes"] if n.task_type_tag != "COMPOSITION"],
            "result": info_node.info["final_result"]["result"]
        }
        return represent
        

    def _collect_inner_graph_infos(self, graph_node, max_dist=100000):
        # return dist_group_precedents, reverse sorted by dist [[dist=k precedents], .., [dist=2 precedents], [dist=1 precedents]]
        need_info_nodes = defaultdict(set)
        existed_need_info_nodes = set()
        
        def get_need_info_nodes(cur, dist):
            if dist > max_dist or (cur.hashkey in existed_need_info_nodes):
                return
            need_info_nodes[dist].add(cur)
            existed_need_info_nodes.add(cur.hashkey)
            for par in sorted(cur.node_graph_info["parent_nodes"], key=lambda x: int(str(x.nid).split(".")[-1])):
                get_need_info_nodes(par, dist+1)

        # Get all predecessor nodes that need information
        for par_graph_node in graph_node.node_graph_info["parent_nodes"]:
            get_need_info_nodes(par_graph_node, 1)
            
        dist_group_precedents = []
        for dist, precedents in sorted(need_info_nodes.items(), reverse=True): # Sort dist from largest to smallest
            precedents = sorted(precedents, key=lambda x: int(str(x.nid).split(".")[-1])) # For parent nodes at the same distance, sort by creation order
            dist_group_precedents.append(precedents)
        return dist_group_precedents
    
   
    def _collect_outer_infos(self, node):
        # return outer infos until the root
        # for outer level dependency, only collect dist=1 inner level dependency
        # [[outer_dist precendents with parent dist = M],]
        outer_dependent_nodes = []
        def get_need_info_nodes(cur, dist):
            if cur is None: return
            outer = cur
            outer_inner_dist_group_precedents = self._collect_inner_graph_infos(outer, max_dist=3)
            if "se a 300-word structured response that: 1) Opens with" in node.task_info["goal"]:
                print("outer_inner_dist_group_precedents", outer_inner_dist_group_precedents, flush=True)
            all_outer_inner_dist_group_precedents = []
            if len(outer_inner_dist_group_precedents) > 0: # has level 1
                for each in outer_inner_dist_group_precedents:
                    all_outer_inner_dist_group_precedents.extend(each)
                outer_inner_dist_group_precedents = all_outer_inner_dist_group_precedents
            outer_dependent_nodes.append(outer_inner_dist_group_precedents)
            get_need_info_nodes(outer.node_graph_info["outer_node"], dist+1)
        
        # get all outer_dependent nodes
        get_need_info_nodes(node.node_graph_info["outer_node"], 1)
        outer_dependent_nodes = outer_dependent_nodes[::-1]
        return outer_dependent_nodes

    
    def collect_node_run_info(self, graph_node):
        """
        为指定的graph_node收集其运行时信息，添加缓存支持
        """
        # 生成缓存键
        cache_key = self._generate_cache_key("node_run_info", graph_node.hashkey, graph_node.nid, graph_node.is_atom)
        
        # 尝试从缓存获取
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for node_run_info: {graph_node.nid}")
            return cached_result
        
        # 缓存未命中，执行原有逻辑
        logger.info(f"Cache miss for node_run_info: {graph_node.nid}, computing...")
            
        if graph_node.is_atom:# For atomic tasks, set the obtained results as its Planning node
            # graph_node = graph_node.node_graph_info["outer_node"].topological_task_queue[0]
            graph_node = graph_node.node_graph_info["outer_node"]
            
        same_graph_precedents = self._collect_inner_graph_infos(graph_node)
        # print("same graph precedents\n{}".format(same_graph_precedents))
        upper_graph_precedents = self._collect_outer_infos(graph_node)

        
        # process
        same_graph_precedents_repre = []
        upper_graph_precedents_repre = []
        for dist_nodes in same_graph_precedents:
            for dist_node in dist_nodes:
                repre = self.get_json_node_info(dist_node)
                if repre is not None:
                    same_graph_precedents_repre.append(repre)
        for dist_nodes in upper_graph_precedents:
            cur_dist_repre = []
            for dist_node in dist_nodes:
                repre = self.get_json_node_info(dist_node)
                if repre is not None:
                    cur_dist_repre.append(repre)
            if len(cur_dist_repre) > 0:
                upper_graph_precedents_repre.append(cur_dist_repre)
            
        result = {
            "same_graph_precedents": same_graph_precedents_repre,
            "upper_graph_precedents": upper_graph_precedents_repre
        }
        
        # 缓存结果
        self._set_to_cache(cache_key, result)
        return result



class InfoNode:
    def __init__(self, hashkey, nid, outer_node, parent_nodes, layer, info):
        self.hashkey = hashkey
        self.nid = nid
        self.outer_node = outer_node
        self.parent_nodes = parent_nodes
        self.layer = layer
        self.info = info