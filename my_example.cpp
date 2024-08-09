#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <algorithm>
#include <cassert>
#include "include/surf.hpp"
#include "bench/bloom.hpp"
#include <set>
#include <algorithm>
#include <chrono>
using namespace surf;

std::random_device rd;

std::vector<std::string> get_indices(const std::vector<std::string>& keys, int page_size, int kv_size) {
    std::vector<std::string> indices;
    int step = page_size / kv_size;
    for (int i = 0; i < keys.size(); i+=step) {
        indices.push_back(keys[i]);
    }
    return indices;
}

int main() {
    std::vector<std::string> keys;
    std::default_random_engine random(rd());
    

    keys.resize(250000);
    for (int i = 0; i < 250000; i++) {
        keys[i].resize(100, 'X');
        for (int j = 0; j < 92; j++) {
            keys[i][j] = random() % 254 + 1;
        }
        
        // unsigned long * pi = (unsigned long *) (keys[i].data() + 92);
        // *pi = i;
        // for (int j = 0; j < 8; j++) {
        //     if (keys[i][92+j] == 0 || keys[i][92+j] == 255) {
        //         keys[i][92+j] = 1;
        //     }
        // }
    }

    // keys.resize(1400000);
    // for (int i = 0; i  < 1400000; i++) {
    //     keys[i].resize(29, 'x');
    //     for (int j = 0; j < 12; j++) {
    //         keys[i][1+j] = random() % 254 + 1;
    //     }
    // }
    auto origin_keys = keys;
    std::sort(keys.begin(), keys.end());
    auto indices = get_indices(keys, 4096, 256);
    std::vector<std::vector<std::string>::iterator> emm;
    std::vector<int> emm2;
    emm.reserve(1);
    emm2.reserve(1);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
        for (const auto & e: origin_keys) {
            emm.push_back(std::lower_bound(indices.begin(), indices.end(), e));
            emm.clear();
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cerr << "ops: " << keys.size() * 10 / elapsed.count() / 1e6 << std::endl;

    SuRF * s = new SuRF(indices, true, 100, surf::kNone, 0, 0);
    
    std::cout << s->getMemoryUsage() << std::endl;
    std::cout << (s->getMemoryUsage()+0.0)/keys.size() << std::endl;
    std::cout << s->louds_dense_->getMemoryUsage() << std::endl;
    std::cout << s->louds_sparse_->getMemoryUsage() << std::endl;
    std::cout << s->louds_sparse_->serializedSize() << std::endl;
    std::cout << s->getHeight() << " " << s->getSparseStartLevel() << std::endl;
    std::set <int> sss;
    
    auto hhh = s->moveToFirst();
    
    while (hhh.isValid()) {
        
        if (hhh.dense_iter_.isComplete()) {
            sss.insert(hhh.dense_iter_.pos_in_trie_.back());
        } else {
            sss.insert(hhh.sparse_iter_.pos_in_trie_[hhh.sparse_iter_.key_len_-1]);
        }
        hhh++;
    }
    auto hit = sss.end();
    --hit;
    std::cerr << (*hit+1.0) << std::endl;
    std::cerr << (*hit+1.0)*16/keys.size() << std::endl;
    std::cerr << ((*hit+1.0)*16 + s->getMemoryUsage())/keys.size() << std::endl;
    
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) {
    for (const auto & e: origin_keys) {
        auto hh = s->moveToKeyLessThan(e, true);
        if (hh.dense_iter_.isComplete()) {
            emm2.push_back(hh.dense_iter_.pos_in_trie_.back());
        } else {
            // std::cerr << "!!" << std::endl;
            // std::cerr << e << " " << hh.getKey() << std::endl;
            // std::cerr << hhh.sparse_iter_.key_len_ << std::endl;
            emm2.push_back(hh.sparse_iter_.pos_in_trie_[hh.sparse_iter_.key_len_-1]);
        }
        emm2.clear();
        // std::cerr << "!" << std::endl;
    }
    }
    t2 = std::chrono::high_resolution_clock::now();
    elapsed = t2 - t1;
    std::cerr << "ops2: " << keys.size() * 10 / elapsed.count() / 1e6 << std::endl;
    
    delete s;

    t1 = std::chrono::high_resolution_clock::now();
    BloomFilter * bf = new BloomFilter(8);
    std::string bf_buf;
    bf->CreateFilter(keys, keys.size(), &bf_buf);
    t2 = std::chrono::high_resolution_clock::now();
    std::cerr << "bloom filter ops: " << keys.size()  / elapsed.count() / 1e6 << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) 
    for (const auto &e: origin_keys) {
        emm2.push_back((int)bf->KeyMayMatch(e, bf_buf));
        assert(emm2.back() == 1);
        emm2.clear();
    }
    t2 = std::chrono::high_resolution_clock::now();
    elapsed = t2 - t1;
    std::cerr << "bloom filter ops: " << keys.size() * 10 / elapsed.count() / 1e6 << std::endl;
    
    delete bf;
    return 0;
}