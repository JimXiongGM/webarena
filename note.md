# note

minimal_example.py 直接调用作者的机器就行，结果不对，都是空的。

步骤：

python scripts/generate_test_data.py

bash prepare.sh

env = ScriptBrowserEnv

obs, info = env.reset

click_action = create_id_based_action(f"click [{match}]")

obs, _, terminated, _, info = env.step(click_action)


evaluator = evaluator_router(config_file)

