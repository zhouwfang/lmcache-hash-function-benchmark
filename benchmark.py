#!/usr/bin/env python3
"""
Benchmark comparing hashlib.sha256() vs xxhash.xxh64() for LMCache's _hash() method
"""

import hashlib
import xxhash
import torch
import numpy as np
import time
import statistics
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from tabulate import tabulate

class HashBenchmark:
    def __init__(self):
        self.results = []
        
    def _hash_sha256(self, tokens: torch.Tensor, prefix_hash: str) -> str:
        """Original implementation using SHA256"""
        hasher = hashlib.sha256()
        hasher.update(prefix_hash.encode("ascii"))
        hasher.update(tokens.numpy().tobytes())
        return hasher.hexdigest()
    
    def _hash_xxh64(self, tokens: torch.Tensor, prefix_hash: str) -> str:
        """New implementation using XXH64"""
        hasher = xxhash.xxh64()
        hasher.update(prefix_hash.encode("ascii"))
        hasher.update(tokens.numpy().tobytes())
        return hasher.hexdigest()
    
    def benchmark_single_hash(self, token_size: int, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark a single token size"""
        # Generate random tokens
        tokens = torch.randint(0, 50000, (token_size,), dtype=torch.int64)
        prefix_hash = "a" * 64  # Typical hash length
        
        # Warm up
        for _ in range(10):
            self._hash_sha256(tokens, prefix_hash)
            self._hash_xxh64(tokens, prefix_hash)
        
        # Benchmark SHA256
        sha256_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self._hash_sha256(tokens, prefix_hash)
            sha256_times.append(time.perf_counter() - start)
        
        # Benchmark XXH64
        xxh64_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self._hash_xxh64(tokens, prefix_hash)
            xxh64_times.append(time.perf_counter() - start)
        
        return {
            'token_size': token_size,
            'sha256_mean': statistics.mean(sha256_times) * 1000,  # Convert to ms
            'sha256_std': statistics.stdev(sha256_times) * 1000,
            'xxh64_mean': statistics.mean(xxh64_times) * 1000,
            'xxh64_std': statistics.stdev(xxh64_times) * 1000,
            'speedup': statistics.mean(sha256_times) / statistics.mean(xxh64_times),
            'sha256_throughput': token_size / statistics.mean(sha256_times) / 1e6,  # M tokens/s
            'xxh64_throughput': token_size / statistics.mean(xxh64_times) / 1e6,
        }
    
    def run_comprehensive_benchmark(self):
        """Run benchmark across different token sizes"""
        # Token sizes to test (typical chunk sizes in LMCache)
        token_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        
        print("Running comprehensive hash benchmark...")
        print("=" * 80)
        
        for size in token_sizes:
            print(f"Benchmarking token size: {size}")
            result = self.benchmark_single_hash(size, iterations=1000)
            self.results.append(result)
        
        self._print_results()
        self._plot_results()
        self._analyze_memory_impact()
    
    def _print_results(self):
        """Print benchmark results in a nice table"""
        headers = ['Token Size', 'SHA256 (ms)', 'XXH64 (ms)', 'Speedup', 
                   'SHA256 (MT/s)', 'XXH64 (MT/s)']
        
        table_data = []
        for r in self.results:
            table_data.append([
                r['token_size'],
                f"{r['sha256_mean']:.4f} ± {r['sha256_std']:.4f}",
                f"{r['xxh64_mean']:.4f} ± {r['xxh64_std']:.4f}",
                f"{r['speedup']:.2f}x",
                f"{r['sha256_throughput']:.2f}",
                f"{r['xxh64_throughput']:.2f}"
            ])
        
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # Summary statistics
        avg_speedup = statistics.mean([r['speedup'] for r in self.results])
        print(f"\nAverage Speedup: {avg_speedup:.2f}x")
        print(f"XXH64 is on average {(avg_speedup-1)*100:.1f}% faster than SHA256")
    
    def _plot_results(self):
        """Create visualization of results"""
        token_sizes = [r['token_size'] for r in self.results]
        sha256_times = [r['sha256_mean'] for r in self.results]
        xxh64_times = [r['xxh64_mean'] for r in self.results]
        speedups = [r['speedup'] for r in self.results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Hash time comparison
        ax1.plot(token_sizes, sha256_times, 'b-o', label='SHA256', linewidth=2)
        ax1.plot(token_sizes, xxh64_times, 'r-o', label='XXH64', linewidth=2)
        ax1.set_xlabel('Token Size')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Hash Computation Time vs Token Size')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup factor
        ax2.plot(token_sizes, speedups, 'g-o', linewidth=2)
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Token Size')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('XXH64 Speedup over SHA256')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hash_benchmark_results.png', dpi=300)
        print("\nPlots saved to 'hash_benchmark_results.png'")
    
    def _analyze_memory_impact(self):
        """Analyze memory usage patterns"""
        print("\n" + "=" * 80)
        print("MEMORY AND SYSTEM IMPACT ANALYSIS")
        print("=" * 80)
        
        # Test with large batch of hashes (simulating real workload)
        num_chunks = 1000
        chunk_size = 1024
        
        print(f"\nSimulating workload: {num_chunks} chunks of {chunk_size} tokens each")
        
        tokens_batch = [torch.randint(0, 50000, (chunk_size,), dtype=torch.int64) 
                       for _ in range(num_chunks)]
        prefix_hashes = ["a" * 64 for _ in range(num_chunks)]
        
        # Batch processing benchmark
        start = time.perf_counter()
        for tokens, prefix in zip(tokens_batch, prefix_hashes):
            self._hash_sha256(tokens, prefix)
        sha256_batch_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for tokens, prefix in zip(tokens_batch, prefix_hashes):
            self._hash_xxh64(tokens, prefix)
        xxh64_batch_time = time.perf_counter() - start
        
        print(f"\nBatch processing time:")
        print(f"  SHA256: {sha256_batch_time:.3f}s")
        print(f"  XXH64:  {xxh64_batch_time:.3f}s")
        print(f"  Speedup: {sha256_batch_time/xxh64_batch_time:.2f}x")

def main():
    print("LMCache Hash Function Performance Benchmark")
    print("Comparing hashlib.sha256() vs xxhash.xxh64()")
    print("=" * 80)
    
    benchmark = HashBenchmark()
    benchmark.run_comprehensive_benchmark()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
