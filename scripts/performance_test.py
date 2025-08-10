#!/usr/bin/env python3
"""
Performance Testing Script for Parallel vs Sequential Processing

This script demonstrates the performance benefits of parallel processing
by comparing the execution time of sequential vs parallel Whisper + NeMo processing.
"""

import asyncio
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
import json

# Mock services for testing (replace with actual imports when ML models are available)
class MockSequentialService:
    """Mock sequential processing service"""
    
    async def process_audio(self, audio_file: str, request: Dict) -> Dict:
        """Simulate sequential processing"""
        start_time = time.time()
        
        # Simulate Whisper processing (2 seconds)
        await asyncio.sleep(2)
        whisper_result = {"text": "Mock Whisper result", "segments": []}
        
        # Simulate NeMo diarization (3 seconds)
        await asyncio.sleep(3)
        diarization_result = {"speakers": ["Speaker_1", "Speaker_2"]}
        
        # Simulate result combination (1 second)
        await asyncio.sleep(1)
        
        processing_time = time.time() - start_time
        return {
            "whisper_result": whisper_result,
            "diarization_result": diarization_result,
            "processing_time": processing_time,
            "method": "sequential"
        }

class MockParallelService:
    """Mock parallel processing service"""
    
    async def process_audio_parallel(self, audio_file: str, request: Dict) -> Dict:
        """Simulate parallel processing"""
        start_time = time.time()
        
        # Run Whisper and NeMo simultaneously
        whisper_task = asyncio.create_task(self._process_whisper())
        diarization_task = asyncio.create_task(self._process_diarization())
        
        # Wait for both to complete
        whisper_result, diarization_result = await asyncio.gather(
            whisper_task, diarization_task
        )
        
        # Simulate result combination (1 second)
        await asyncio.sleep(1)
        
        processing_time = time.time() - start_time
        return {
            "whisper_result": whisper_result,
            "diarization_result": diarization_result,
            "processing_time": processing_time,
            "method": "parallel"
        }
    
    async def _process_whisper(self) -> Dict:
        """Simulate Whisper processing"""
        await asyncio.sleep(2)
        return {"text": "Mock Whisper result", "segments": []}
    
    async def _process_diarization(self) -> Dict:
        """Simulate NeMo diarization"""
        await asyncio.sleep(3)
        return {"speakers": ["Speaker_1", "Speaker_2"]}

async def run_performance_test(
    num_runs: int = 5,
    num_files: int = 3
) -> Dict[str, Any]:
    """
    Run performance comparison test
    
    Args:
        num_runs: Number of test runs for averaging
        num_files: Number of files to process in each run
    
    Returns:
        Performance comparison results
    """
    print(f"🚀 Starting Performance Test")
    print(f"   Runs: {num_runs}")
    print(f"   Files per run: {num_files}")
    print(f"   Total files to process: {num_runs * num_files}")
    print()
    
    # Initialize services
    sequential_service = MockSequentialService()
    parallel_service = MockParallelService()
    
    # Test results storage
    sequential_times = []
    parallel_times = []
    
    for run in range(num_runs):
        print(f"📊 Run {run + 1}/{num_runs}")
        
        # Sequential processing
        sequential_start = time.time()
        for file_num in range(num_files):
            result = await sequential_service.process_audio(
                f"test_file_{file_num}.wav",
                {"language": "en"}
            )
        sequential_total = time.time() - sequential_start
        sequential_times.append(sequential_total)
        
        # Parallel processing
        parallel_start = time.time()
        tasks = []
        for file_num in range(num_files):
            task = parallel_service.process_audio_parallel(
                f"test_file_{file_num}.wav",
                {"language": "en"}
            )
            tasks.append(task)
        
        # Process all files in parallel
        results = await asyncio.gather(*tasks)
        parallel_total = time.time() - parallel_start
        parallel_times.append(parallel_total)
        
        print(f"   Sequential: {sequential_total:.2f}s")
        print(f"   Parallel:   {parallel_total:.2f}s")
        print(f"   Speedup:    {sequential_total/parallel_total:.2f}x")
        print()
    
    # Calculate statistics
    sequential_avg = statistics.mean(sequential_times)
    parallel_avg = statistics.mean(parallel_times)
    speedup_avg = sequential_avg / parallel_avg
    
    sequential_std = statistics.stdev(sequential_times) if len(sequential_times) > 1 else 0
    parallel_std = statistics.stdev(parallel_times) if len(parallel_times) > 1 else 0
    
    # Calculate theoretical vs actual speedup
    # Sequential: Whisper(2s) + NeMo(3s) + Combine(1s) = 6s per file
    # Parallel: max(Whisper(2s), NeMo(3s)) + Combine(1s) = 4s per file
    theoretical_speedup = 6.0 / 4.0  # 1.5x per file
    
    # For multiple files, parallel processing should be even better
    theoretical_multi_file_speedup = (6.0 * num_files) / (4.0 + (num_files - 1) * 0.1)
    
    results = {
        "test_config": {
            "num_runs": num_runs,
            "num_files_per_run": num_files,
            "total_files": num_runs * num_files
        },
        "sequential_processing": {
            "average_time": sequential_avg,
            "std_deviation": sequential_std,
            "times": sequential_times
        },
        "parallel_processing": {
            "average_time": parallel_avg,
            "std_deviation": parallel_std,
            "times": parallel_times
        },
        "performance_comparison": {
            "average_speedup": speedup_avg,
            "theoretical_speedup_per_file": theoretical_speedup,
            "theoretical_multi_file_speedup": theoretical_multi_file_speedup,
            "efficiency": (speedup_avg / theoretical_multi_file_speedup) * 100
        },
        "analysis": {
            "sequential_bottleneck": "Whisper and NeMo run sequentially",
            "parallel_advantage": "Whisper and NeMo run simultaneously",
            "scaling_benefit": "Parallel processing scales better with multiple files",
            "gpu_utilization": "Better GPU memory and compute utilization"
        }
    }
    
    return results

def print_results(results: Dict[str, Any]):
    """Print formatted performance test results"""
    print("=" * 60)
    print("📈 PERFORMANCE TEST RESULTS")
    print("=" * 60)
    
    config = results["test_config"]
    print(f"Test Configuration:")
    print(f"  • Runs: {config['num_runs']}")
    print(f"  • Files per run: {config['num_files_per_run']}")
    print(f"  • Total files processed: {config['total_files']}")
    print()
    
    seq = results["sequential_processing"]
    par = results["parallel_processing"]
    comp = results["performance_comparison"]
    
    print(f"Sequential Processing:")
    print(f"  • Average time: {seq['average_time']:.2f}s ± {seq['std_deviation']:.2f}s")
    print(f"  • Individual times: {[f'{t:.2f}s' for t in seq['times']]}")
    print()
    
    print(f"Parallel Processing:")
    print(f"  • Average time: {par['average_time']:.2f}s ± {par['std_deviation']:.2f}s")
    print(f"  • Individual times: {[f'{t:.2f}s' for t in par['times']]}")
    print()
    
    print(f"Performance Comparison:")
    print(f"  • Average speedup: {comp['average_speedup']:.2f}x")
    print(f"  • Theoretical speedup (per file): {comp['theoretical_speedup_per_file']:.2f}x")
    print(f"  • Theoretical speedup (multi-file): {comp['theoretical_multi_file_speedup']:.2f}x")
    print(f"  • Efficiency: {comp['efficiency']:.1f}%")
    print()
    
    print(f"Analysis:")
    for key, value in results["analysis"].items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    print()
    
    if comp['average_speedup'] > 1.0:
        print("✅ Parallel processing provides significant performance improvement!")
        if comp['average_speedup'] > 1.5:
            print("🚀 Excellent scaling with parallel processing!")
        elif comp['average_speedup'] > 1.2:
            print("📈 Good performance improvement with parallel processing!")
    else:
        print("⚠️  Parallel processing did not show improvement in this test")
    
    print("=" * 60)

async def main():
    """Main function to run performance tests"""
    print("🎯 Enhanced Whisper Diarization Service - Performance Test")
    print("=" * 60)
    print()
    
    try:
        # Run performance test
        results = await run_performance_test(num_runs=5, num_files=3)
        
        # Print results
        print_results(results)
        
        # Save results to file
        output_file = Path("outputs/performance_test_results.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
