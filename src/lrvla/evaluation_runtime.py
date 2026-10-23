"""Optional generated-client features, keeping the pinned upstream tree intact."""
from __future__ import annotations

import ast


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Evaluation runtime anchor changed: {old[:80]!r}")
    return source.replace(old, new, 1)


def add_runtime_features(source, *, scene_rng=False, expert_cache=False, timing=False):
    if scene_rng:
        source = replace_once(source, '    expert_check = True\n',
                              '    from policy_sessions import scene_seed\n    expert_check = True\n')
        source = replace_once(source,
                              "        ret = model.infer(dict(reset=True, robo_name=usr_args['robo_name']))",
                              "        ret = model.infer(dict(reset=True, robo_name=usr_args['robo_name'], "
                              "rng_seed=scene_seed(task_name, args['task_config'], now_seed)))")
        source = replace_once(source, "    model = WebsocketClientPolicy(port=usr_args['port'])\n",
                              "    model = WebsocketClientPolicy(port=usr_args['port'])\n"
                              "    from policy_sessions import POLICY_RNG\n"
                              "    if model.get_server_metadata().get('policy_rng') != POLICY_RNG:\n"
                              "        raise RuntimeError('Inference service lacks the requested scene RNG protocol')\n")
    if expert_cache:
        source = replace_once(source, '    expert_check = True\n',
                              '    from scene_bank import SceneBank\n'
                              '    bank = SceneBank.from_environment(task_name, args["task_config"], test_num, st_seed)\n'
                              '    expert_check = True\n')
        start = source.index("        episode_info = None\n")
        end = source.index("        succ_seed += 1\n", start)
        expert = source[start:end]
        expert = expert.replace("        episode_info = None\n", "        episode_info = None\n        attempts = []\n")
        expert = expert.replace("        for expert_attempt in range(5):\n",
                                "        for expert_attempt in range(5):\n            expert_error = None\n")
        expert = expert.replace("            except Exception as error:\n",
                                "            except Exception as error:\n                expert_error = str(error)[:1000]\n")
        expert = expert.replace("            if expert_success:\n",
                                '            attempts.append({"seed": now_seed, "success": bool(expert_success), "error": expert_error})\n'
                                "            if expert_success:\n")
        replacement = '''        cache_start_seed = now_seed
        cached_scene = bank.lookup(now_id, now_seed)
        if cached_scene is not None:
            # Match the original first expert initialization's robot/planner
            # warmup, then rebuild the actual policy scene below as usual.
            if now_id == 0:
                try:
                    TASK_ENV.setup_demo(now_ep_num=now_id, seed=cached_scene["attempts"][0]["seed"], is_test=True, **args)
                except Exception as error:
                    print(f"Cache replay planner warmup: {error}")
                finally:
                    TASK_ENV.close_env()
            now_seed = cached_scene["policy_seed"]
            episode_info = cached_scene["episode_info"]
        else:
'''
        replacement += "".join("    " + line if line.strip() else line for line in expert.splitlines(keepends=True))
        replacement += '''            cached_scene = bank.expert_done(now_id, cache_start_seed, now_seed, {"info": episode_info["info"]}, attempts)
        bank.before_setup(cached_scene)
        cache_first_observation = True
'''
        source = source[:start] + replacement + source[end:]
        source = replace_once(source, '        scene_info = episode_info["info"]',
                              '        bank.before_instruction(cached_scene)\n        scene_info = episode_info["info"]')
        source = replace_once(source, '        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction\n',
                              '        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction\n'
                              '        bank.instruction(cached_scene, instruction)\n')
        source = replace_once(source, '            observation = TASK_ENV.get_obs()\n',
                              '            observation = TASK_ENV.get_obs()\n'
                              '            if cache_first_observation:\n'
                              '                bank.first_observation(cached_scene, observation)\n'
                              '                cache_first_observation = False\n')
        source = replace_once(source, '        except UnStableError as error:\n',
                              '        except UnStableError as error:\n            bank.unstable(cached_scene)\n')
    if timing:
        source = replace_once(source, '    expert_check = True\n',
                              '    from episode_metrics import EpisodeMetrics\n    expert_check = True\n')
        source = replace_once(source, '    while succ_seed < test_num:\n',
                              '    while succ_seed < test_num:\n'
                              '        metrics = EpisodeMetrics(task_name, args["task_config"], now_id, now_seed)\n'
                              '        metrics.phase_start("expert_seconds")\n')
        source = replace_once(source, '        try:\n            TASK_ENV.setup_demo',
                              '        metrics.phase_start("policy_setup_seconds")\n        try:\n            TASK_ENV.setup_demo')
        source = replace_once(source, '        scene_info = episode_info["info"]',
                              '        metrics.phase_start("instruction_seconds")\n        scene_info = episode_info["info"]')
        source = replace_once(source, '        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction\n',
                              '        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction\n'
                              '        metrics.phase_start(None)\n')
        source = replace_once(source, '            observation = TASK_ENV.get_obs()\n',
                              '            with metrics.measure("observation_seconds"):\n                observation = TASK_ENV.get_obs()\n')
        source = replace_once(source, '            ret = model.infer(formatted_observation) #(TASK_ENV, model, observation)\n',
                              '            with metrics.measure("rpc_seconds"):\n'
                              '                ret = model.infer(formatted_observation)\n'
                              '            metrics.inference(ret.get("server_timing"))\n')
        start = source.index('            if len(action.shape) == 2:\n')
        end = source.index('            print(f"infer time {latency}")', start)
        source = (source[:start] + '            with metrics.measure("action_seconds"):\n'
                  + "".join("    " + line if line.strip() else line for line in source[start:end].splitlines(keepends=True))
                  + source[end:])
        source = replace_once(source, '            # A scene that cannot initialize still consumes its policy trial.\n',
                              '            metrics.finish(seed=now_seed, success=False, initialization_failed=True)\n'
                              '            # A scene that cannot initialize still consumes its policy trial.\n')
        source = replace_once(source, '\n        TASK_ENV.test_num += 1\n',
                              '\n        TASK_ENV.test_num += 1\n        metrics.finish(seed=now_seed, success=succ)\n')
    # A bind-all address is not a client destination; both processes run on this node.
    source = replace_once(source, "    model = WebsocketClientPolicy(port=usr_args['port'])\n",
                          "    model = WebsocketClientPolicy(host='127.0.0.1', port=usr_args['port'])\n")
    tree = ast.parse(source)
    if expert_cache:
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "eval_policy")
        assignment = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "bank" for target in node.targets))
        closing = ast.parse("bank.close()").body
        function.body[assignment + 1:] = [ast.Try(body=function.body[assignment + 1:], handlers=[],
                                                orelse=[], finalbody=closing)]
        source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    return source
