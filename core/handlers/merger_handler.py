from core.pipeline_core.pipeline_core import DataHandler, PipelineContext
from loguru import logger
import pandas as pd

class MergerHandler(DataHandler):
    """
    Merge context dataframe
    Logs process in the context
    """
    def process(self, context: PipelineContext) -> PipelineContext:
        if not context.data_map:
            raise ValueError("❌ MergerHandler : data_map is empty. Nothing to merge")

        source_names = list(context.data_map.keys())
        logger.info(f"🔄 Merge sources : {source_names}")

        # 1. Check for columns consistance
        first_df_cols = set(context.data_map[source_names[0]].columns)
        for name in source_names[1:]:
            current_cols = set(context.data_map[name].columns)
            if first_df_cols != current_cols:
                diff = first_df_cols.symmetric_difference(current_cols)
                logger.warning(f"⚠️ Diffrence between columns was detected {name}: {diff}")
                # Check if we can merge columuns 

        # 2. Prepare and merge
        frames_to_concat = []
        for name, df in context.data_map.items():
            temp_df = df.copy()
            temp_df['source_origin'] = name  # Ajout de la provenance
            frames_to_concat.append(temp_df)

        merged_df = pd.concat(frames_to_concat, ignore_index=True)
        initial_count = len(merged_df)

        # 3. Duplicates handling
        # 'source_origin' ignored to identify real business duplicates
        subset_cols = [col for col in merged_df.columns if col != 'source_origin']
        context.final_df = merged_df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
        
        duplicates_removed = initial_count - len(context.final_df)

        # 4. Store logs in context (dict metadata/logs)
        context.metadata['merger_report'] = {
            'initial_rows': initial_count,
            'final_rows': len(context.final_df),
            'duplicates_removed': duplicates_removed,
            'sources': source_names
        }

        logger.success(f"✅ Merge complete: {len(context.final_df)} rows kept ({duplicates_removed} duplicates remove).")
        
        return context