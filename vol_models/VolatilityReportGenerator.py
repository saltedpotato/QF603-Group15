from vol_models.model_load_packages import *

class VolatilityReportGeneratorStats:
    """
    A comprehensive report generator for volatility forecasting analysis.
    Saves plots and outputs in structured markdown format for presentation.
    """
    
    def __init__(self, report_name="volatility_forecast_report"):
        self.report_name = report_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.base_dir = Path("./report_output_v6")
        self.images_dir = self.base_dir / "images"
        self.base_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        
        # Report file
        self.report_file = self.base_dir / f"{report_name}_{self.timestamp}.md"
        
        # Initialize report
        self._init_report()
        
    def _init_report(self):
        """Initialize the markdown report with title and TOC"""
        with open(self.report_file, 'w') as f:
            f.write(f"# Volatility Forecasting Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Author:** PhD Research Team\n\n")
            f.write(f"---\n\n")
            f.write(f"## Table of Contents\n\n")
            f.write(f"1. [Executive Summary](#executive-summary)\n")
            f.write(f"2. [Data Description](#data-description)\n")
            f.write(f"3. [Methodology](#methodology)\n")
            f.write(f"4. [Volatility Estimators Analysis](#volatility-estimators-analysis)\n")
            f.write(f"5. [HAR Model Results](#har-model-results)\n")
            f.write(f"6. [HAR-X Model Results](#har-x-model-results)\n")
            f.write(f"7. [Model Comparison](#model-comparison)\n")
            f.write(f"8. [Test Set Evaluation](#test-set-evaluation)\n")
            f.write(f"9. [Conclusions](#conclusions)\n")
            f.write(f"10. [Appendix](#appendix)\n\n")
            f.write(f"---\n\n")
    
    def add_section(self, title, level=2):
        """Add a section header"""
        with open(self.report_file, 'a') as f:
            f.write(f"\n{'#' * level} {title}\n\n")
    
    def add_text(self, text):
        """Add text content"""
        with open(self.report_file, 'a') as f:
            f.write(f"{text}\n\n")
    
    def add_table(self, df, caption=""):
        """Add a pandas DataFrame as markdown table"""
        with open(self.report_file, 'a') as f:
            if caption:
                f.write(f"**{caption}**\n\n")
            f.write(df.to_markdown())
            f.write("\n\n")
    
    def save_and_add_plot(self, fig, filename, caption="", width=800):
        """Save matplotlib figure and add to report"""
        # Save figure
        img_path = self.images_dir / f"{filename}.png"
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        
        # Add to report
        with open(self.report_file, 'a') as f:
            if caption:
                f.write(f"**{caption}**\n\n")
            f.write(f"![{caption}](images/{filename}.png)\n\n")
        
        print(f"✓ Saved plot: {filename}.png")
        return str(img_path)
    
    def add_metrics_summary(self, metrics_dict, title="Metrics Summary"):
        """Add metrics in a formatted way"""
        with open(self.report_file, 'a') as f:
            f.write(f"**{title}**\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    f.write(f"| {key} | {value:.4f} |\n")
                else:
                    f.write(f"| {key} | {value} |\n")
            f.write("\n\n")
    
    def add_code_output(self, output, title=""):
        """Add code output in formatted code block"""
        with open(self.report_file, 'a') as f:
            if title:
                f.write(f"**{title}**\n\n")
            f.write("```\n")
            f.write(str(output))
            f.write("\n```\n\n")
    
    def finalize_report(self):
        """Finalize the report"""
        with open(self.report_file, 'a') as f:
            f.write(f"\n---\n\n")
            f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n")
        
        print(f"\n{'='*60}")
        print(f"✓ Report generated successfully!")
        print(f"  Location: {self.report_file}")
        print(f"  Images:   {self.images_dir}")
        print(f"{'='*60}\n")

class VolatilityReportGeneratorML:
    """
    A comprehensive report generator for volatility forecasting analysis.
    Saves plots and outputs in structured markdown format.
    Supports sequential model runs with automatic appending and TOC updates.
    """
    
    def __init__(self, report_name="ml_tft_report", append=False, find_latest=True):
        """
        Initialize the report generator.
        
        Parameters:
        -----------
        report_name : str
            Base name for the report file
        append : bool
            If True, find and append to the latest report file
        find_latest : bool
            If True and append=False, still look for latest report to continue
        """
        self.report_name = report_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_folder = Path(f"report_output_v6")
        self.report_folder.mkdir(exist_ok=True)
        self.image_folder = self.report_folder / "images"
        self.image_folder.mkdir(exist_ok=True)
        self.run_number = 1
        self.is_appending = False
        
        # Try to find and append to latest report
        report_files = sorted(self.report_folder.glob(f"{self.report_name}_*.md"), reverse=True)
        
        if (append or find_latest) and report_files:
            self.report_file = report_files[0]
            self.is_appending = True
            print(f"✓ Appending to existing report: {self.report_file}")
            
            # Read existing content
            with open(self.report_file, 'r') as f:
                self.report_content = f.read()
            
            # Count existing model runs to determine run number
            self.run_number = self.report_content.count("## Model Run") + 1
            
            # Add separator and new run header
            self._append_new_run()
        else:
            # Create a new report
            self.report_file = self.report_folder / f"{self.report_name}_{self.timestamp}.md"
            self.report_content = ""
            self._init_report()
        
    def _init_report(self):
        """Initialize a new markdown report"""
        report_text = f"""# ML & TFT Models Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Table of Contents

1. [Model Run 1](#model-run-1)

---

"""
        with open(self.report_file, 'w') as f:
            f.write(report_text)
        self.report_content = report_text
        self.add_section(f"Model Run {self.run_number}", level=2)
    
    def _append_new_run(self):
        """Append a new model run section to existing report"""
        # Write updated content to file
        with open(self.report_file, 'w') as f:
            f.write(self.report_content)
        
        # Add new run section
        new_run_text = f"\n## Model Run {self.run_number}\n\n"
        with open(self.report_file, 'a') as f:
            f.write(new_run_text)
        
        # Update in-memory content
        self.report_content += new_run_text
        print(f"✓ Started Model Run {self.run_number}")
    
    def add_section(self, title, level=2):
        """Add a section heading"""
        section_text = f"\n{'#' * level} {title}\n\n"
        with open(self.report_file, 'a') as f:
            f.write(section_text)
        self.report_content += section_text
    
    def add_text(self, text):
        """Add text content"""
        text_block = f"{text}\n\n"
        with open(self.report_file, 'a') as f:
            f.write(text_block)
        self.report_content += text_block
    
    def add_table(self, df, caption=""):
        """Add a table in markdown format"""
        table_text = ""
        if caption:
            table_text += f"**{caption}**\n\n"
        table_text += df.to_markdown() + "\n\n"
        
        with open(self.report_file, 'a') as f:
            f.write(table_text)
        self.report_content += table_text
    
    def add_metrics_summary(self, metrics_dict, title="Metrics Summary"):
        """Add metrics as a formatted table"""
        df = pd.DataFrame(metrics_dict, index=[0]).T
        df.columns = ['Value']
        self.add_table(df, caption=title)
    
    def save_and_add_plot(self, fig, filename, caption=""):
        """Save plot and add to report"""
        # Create timestamped filename to avoid conflicts across runs
        if self.run_number > 1:
            timestamped_filename = f"{filename}_run{self.run_number}"
        else:
            timestamped_filename = filename
            
        # Save plot
        plot_path = self.image_folder / f"{timestamped_filename}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        # Add to report
        plot_text = ""
        if caption:
            plot_text += f"**{caption}**\n\n"
        plot_text += f"![{caption}](images/{timestamped_filename}.png)\n\n"
        
        with open(self.report_file, 'a') as f:
            f.write(plot_text)
        self.report_content += plot_text
        
        print(f"  ✓ Saved plot: {timestamped_filename}.png")
    
    def add_run_summary(self, model_name, metrics_dict):
        """Add a summary for the completed model run"""
        summary = f"""
### {model_name} Summary

**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        with open(self.report_file, 'a') as f:
            f.write(summary)
        self.report_content += summary
        
        self.add_metrics_summary(metrics_dict, title=f"{model_name} Performance Metrics")
    
    def finalize_report(self, final_message="Report generation completed"):
        """Finalize the report with a closing message"""
        closing = f"""
---

**{final_message}**

*Last Updated: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*
"""
        with open(self.report_file, 'a') as f:
            f.write(closing)
        
        print(f"\n{'='*60}")
        print(f"✓ Report saved to: {self.report_file}")
        print(f"  Images:   {self.image_folder}")
        print(f"  Model Runs: {self.run_number}")
        print(f"{'='*60}\n")