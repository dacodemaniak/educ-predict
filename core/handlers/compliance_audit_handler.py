import mlflow
import os
from datetime import datetime
from core.pipeline_core.pipeline_core import DataHandler, PipelineContext
from loguru import logger

class ComplianceAuditHandler(DataHandler):
    """
    Build RGPD/AIAct compliance report and log it in MLFlow
    """
    def __init__(self, output_dir: str = "outputs"):
        super().__init__()
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.final_df is None:
            return context

        logger.info("📄 Build compliance report...")
        
        report_name = f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        report_path = os.path.join(self.output_dir, report_name)
        
        df = context.final_df
        sensitive_cols = ["romantic", "Dalc", "Walc", "health"]
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== COMPLIANCE REPORT (RGPD & AI ACT) ===\n")
            f.write(f"Date de l'audit : {datetime.now().isoformat()}\n\n")
            
            # 1. Check data sensitivity (RGPD Art. 9)
            f.write("[1] CHECK DATA SENSITIVITY\n")
            found_sensitive = [c for c in sensitive_cols if c in df.columns]
            if not found_sensitive:
                f.write("✅ SUCCESS : No Health or Behavior datas was found.\n")
            else:
                f.write(f"❌ ALERT : Sensitive datas was found : {found_sensitive}\n")
            
            # 2. Anonymization check (AI Act)
            f.write("\n[2] ANONYMIZATION CHECK\n")
            if 'age' in df.columns:
                max_age = df['age'].max()
                f.write(f"- Max age detected : {max_age} (Awaited threshold: 19)\n")
            if 'school' in df.columns:
                is_hidden = "HIDDEN" in df['school'].values
                f.write(f"- School hiding : {'✅ Applied' if is_hidden else '❌ Not found'}\n")

            # 3. Minimization stats
            f.write("\n[3] MINIMIZATION STATS\n")
            f.write(f"- Total features after encoding : {len(df.columns)}\n")
            f.write(f"- Audited rows : {len(df)}\n")

        # Log artifact with MLFlow
        try:
            with mlflow.start_run(run_name=f"Compliance_Audit_{datetime.now().strftime('%Y%m%d')}"):
                mlflow.set_tag("type", "compliance_audit")
                mlflow.set_tag("version", "1.0")

                mlflow.log_artifact(report_path)

                mlflow.log_metric("rows_audited", len(context.final_df))
                if mlflow.active_run() is not None:
                    logger.success(f"✅ Compliance report logged in MLFlow (Run: {mlflow.active_run().info.run_id})")
        except Exception as e:
            logger.error(f"❌ Failed to log Compliance report to MLFlow : {e}")

        # Save context to log
        context.metadata['compliance_report_path'] = report_path
        logger.success(f"✅ Compliance report generated : {report_path}")
        
        return context