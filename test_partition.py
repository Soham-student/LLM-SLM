from system_detector import detect_system
from model_catalog import get_model
from model_partitioner import partition_model, summarize_plan

profile = detect_system()

# Test multiple models
for key in ['qwen2.5-1.5b', 'qwen2.5-3b', 'mistral-7b', 'llama-3.1-70b']:
    model = get_model(key)
    plan = partition_model(model, profile)
    print("=" * 60)
    print(summarize_plan(plan))
    print()
