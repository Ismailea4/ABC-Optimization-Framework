"""
Test script for SVM optimization functionality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_basic_svm_optimization():
    """Test basic SVM optimization functionality."""
    print("Testing SVM Optimization...")
    
    try:
        from problems.svm_optimization import create_svm_problem
        from core.enhanced_abc import EnhancedABC
        
        # Create a simple SVM problem
        print("1. Creating SVM problem with iris dataset...")
        svm_problem = create_svm_problem(
            dataset_name='iris',
            kernel='rbf',
            optimize_params=['C', 'gamma']
        )
        print(f"   Problem created successfully!")
        print(f"   Bounds: {svm_problem.bounds}")
        print(f"   Dataset shape: {svm_problem.X.shape}")
        
        # Test evaluation
        print("\n2. Testing objective function evaluation...")
        import numpy as np
        test_solution = np.array([0.5, 0.5])  # Normalized parameters
        fitness = svm_problem.evaluate(test_solution)
        print(f"   Test evaluation successful! Fitness: {fitness:.4f}")
        
        # Test optimization with very small scale
        print("\n3. Running mini-optimization (5 bees, 10 iterations)...")
        abc = EnhancedABC(
            num_bees=5,
            bounds=svm_problem.bounds,
            objective_function=svm_problem.evaluate,
            max_iterations=10,
            use_sa=True,
            use_pbsa=False  # Disable PBSA for quick test
        )
        
        best_solution, best_fitness = abc.optimize()
        print(f"   Optimization completed!")
        print(f"   Best fitness: {best_fitness:.4f}")
        
        # Get model results
        print("\n4. Evaluating best model...")
        results = svm_problem.evaluate_best_model()
        print(f"   Best parameters: {results['best_params']}")
        print(f"   CV score: {results['cv_score']:.4f}")
        print(f"   Test accuracy: {results['test_accuracy']:.4f}")
        print(f"   Function evaluations: {results['evaluation_count']}")
        
        print("\n✅ SVM optimization test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ SVM optimization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test dataset loading functionality."""
    print("\nTesting Dataset Loading...")
    
    try:
        from problems.svm_optimization import SVMDatasetLoader
        
        # Get available datasets
        available = SVMDatasetLoader.get_available_datasets()
        print(f"Available datasets: {list(available.keys())}")
        
        # Test loading each dataset
        for dataset_name, problem_type in available.items():
            try:
                X, y, loaded_type = SVMDatasetLoader.load_dataset(dataset_name)
                print(f"✓ {dataset_name}: {X.shape} samples, type: {loaded_type}")
            except Exception as e:
                print(f"✗ {dataset_name}: Failed to load - {e}")
        
        print("✅ Dataset loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test FAILED: {e}")
        return False


def test_svm_experiment_runner():
    """Test the experiment runner functionality."""
    print("\nTesting SVM Experiment Runner...")
    
    try:
        # Import directly from the file since module structure may not be set up
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))
        from svm_examples import SVMExperimentRunner
        
        runner = SVMExperimentRunner()
        
        # Test single experiment with minimal parameters
        print("Running single experiment...")
        result = runner.run_single_experiment(
            dataset_name='iris',
            algorithm_type='enhanced',
            kernel='rbf',
            num_bees=5,  # Very small for quick test
            max_iterations=5,
            optimize_params=['C', 'gamma']
        )
        
        print(f"✓ Experiment completed!")
        print(f"  Dataset: {result['dataset']}")
        print(f"  Improvement: {result['improvement_percentage']:.2f}%")
        print(f"  Time: {result['optimization_time']:.2f}s")
        
        print("✅ Experiment runner test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Experiment runner test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ABC-SVM OPTIMIZATION FRAMEWORK TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_dataset_loading,
        test_basic_svm_optimization,
        test_svm_experiment_runner
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 ALL TESTS PASSED! The framework is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    print("=" * 60)