import yaml
import importlib
from loguru import logger

class PipelineBuilder:
    @staticmethod
    def build_from_yaml(config_path: str, orchestrator):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.info(f"🏗️ Building pipeline from Notebook classes...")

        # 1. Data processing steps
        for step_cfg in config['pipeline']['data_processing']:
            handler_name = step_cfg['step']
            params = step_cfg.get('params', {})
            
            handler = PipelineBuilder._instantiate_notebook_handler(handler_name, step_cfg, params)
            if handler:
                orchestrator.add_handler(handler)

        # 2. Learning steps
        learning_cfg = config['pipeline'].get('learning', {})
        strategies = learning_cfg.get('strategies', [])
        scenarii = learning_cfg.get('scenarii', [])

        for scenario in scenarii:
            for strat_name in strategies:
                # Looking for strategy in the Notebook or imported files
                strategy = PipelineBuilder._get_class_from_anywhere(
                    strat_name, 
                    "core.strategy_core.training_strategies"
                )(scenario_id=scenario['label'], exclusions=scenario['exclusions'])
                
                # ModelHandler is supposed to be defined in the Notebook or pipeline_core
                ModelHandlerClass = PipelineBuilder._get_class_from_anywhere("ModelHandler", "core.handlers.model_handler")
                handler = ModelHandlerClass(strategy=strategy, scenario_label=scenario['label'])
                orchestrator.add_handler(handler)

        logger.success("✅ Pipeline dynamically orchestrated from the Notebook.")
        return orchestrator

    @staticmethod
    def _get_class_from_anywhere(class_name, module_path=None):
        """Look for a class in the Notebook (globals) or in a specific module"""
        # 1. Look in the classes defined in the Notebook
        if class_name in globals():
            return globals()[class_name]
        
        # 2. If not found and a module path is provided, import the .py file
        if module_path:
            logger.debug(f"🔍 Looking for class '{class_name}' in module '{module_path}'")
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        
        raise AttributeError(f"❌ Class '{class_name}' is not found in the Notebook or the module {module_path}.")

    @staticmethod
    def _instantiate_notebook_handler(name, config, params):
        """Specific instantiation for Notebook Handlers"""

        # Special cases of Handlers requiring an internal strategy
        if name == "OutlierHandler":
            target_cols = params.get('target_columns', [])
            strategy_params = {k: v for k, v in params.items() if k != 'target_columns'}
            StrategyClass = PipelineBuilder._get_class_from_anywhere(
                config['strategy'], 
                "core.strategy_core.outliers_strategies"
            )
            strat = StrategyClass(**strategy_params)
            HandlerClass = PipelineBuilder._get_class_from_anywhere("OutlierHandler", "core.handlers.outlier_handler")
            return HandlerClass(strategy=strat, target_columns=target_cols)
        
        if name == "ImputationHandler":
            StrategyClass = PipelineBuilder._get_class_from_anywhere(
                config['strategy'], 
                "core.strategy_core.imputation_strategies"
            )
            strat = StrategyClass(**params)
            HandlerClass = PipelineBuilder._get_class_from_anywhere("ImputationHandler", "core.handlers.imputation_handler")
            return HandlerClass(strategy=strat)

        # General case: retrieve the class from the Notebook and instantiate it
        HandlerClass = PipelineBuilder._get_class_from_anywhere(class_name=name, module_path="core.handlers."+PipelineBuilder._to_snake_case(name))
        return HandlerClass(**params)
    
    @staticmethod
    def _to_snake_case(name: str) -> str:
        import re
        partial_s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        snake_s = re.sub('([a-z0-9])([A-Z])', r'\1_\2', partial_s)
        return snake_s.lower()