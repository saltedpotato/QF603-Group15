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
    """
    
    def __init__(self, report_name="ml_tft_report", append=False):
        self.report_name = report_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_folder = Path(f"report_output_v6")
        self.report_folder.mkdir(exist_ok=True)
        self.image_folder = self.report_folder / "images"
        self.image_folder.mkdir(exist_ok=True)
        
        # Find the latest report if in append mode
        if append:
            report_files = sorted(self.report_folder.glob(f"{self.report_name}_*.md"), reverse=True)
            if report_files:
                self.report_file = report_files[0]
                print(f"Appending to existing report: {self.report_file}")
                with open(self.report_file, 'r') as f:
                    self.report_content = f.read()

                # Define the new TOC entries
                new_toc_entries = [
                    "7. [Machine Learning Models Results](#machine-learning-models-results)",
                    "8. [Temporal Fusion Transformer (TFT) Results](#temporal-fusion-transformer-tft-results)",
                    "9. [Comprehensive Model Comparison](#comprehensive-model-comparison)"
                ]
                
                # Find the position to insert the new entries (before Conclusions)
                toc_lines = self.report_content.split('## Table of Contents')[1].split('---')[0].splitlines()
                
                # Filter out old entries that will be replaced/renumbered
                existing_entries = [line for line in toc_lines if line.strip() and not any(new_entry.split('](')[0] in line for new_entry in new_toc_entries)]
                
                # Find insertion point
                insertion_point = -1
                for i, line in enumerate(existing_entries):
                    if "conclusions" in line.lower():
                        insertion_point = i
                        break
                if insertion_point == -1:
                    insertion_point = len(existing_entries) -1 # Fallback to before appendix

                # Combine and renumber
                final_toc_list = existing_entries[:insertion_point] + new_toc_entries + existing_entries[insertion_point:]
                
                # Renumber the whole list
                renumbered_toc = []
                for i, line in enumerate(final_toc_list):
                    if line.strip().startswith(tuple(f"{j}." for j in range(20))):
                        parts = line.split('.', 1)
                        renumbered_toc.append(f"{i}.{parts[1]}")

                # Reconstruct the full TOC string
                new_toc_section = "## Table of Contents\n" + "\n".join(renumbered_toc) + "\n---\n"

                # Replace the old TOC in the report content
                start_marker = "## Table of Contents"
                end_marker = "---"
                start_index = self.report_content.find(start_marker)
                end_index = self.report_content.find(end_marker, start_index)
                
                if start_index != -1 and end_index != -1:
                    self.report_content = self.report_content[:start_index] + new_toc_section + self.report_content[end_index + len(end_marker):]

                # Add a separator for the new run and convert back to list of lines
                self.report_content = self.report_content.splitlines(keepends=True)
                self.add_section(f"New Analysis Run - {self.timestamp}", level=1)
                return

        # If not appending or no file found, create a new one
        self.report_file = self.report_folder / f"{self.report_name}_{self.timestamp}.md"
        self.report_content = []
        self._init_report()
        
    def _init_report(self):
        """Initialize the markdown report with title"""
        with open(self.report_file, 'w') as f:
            f.write(f"# ML & TFT Models Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"---\n\n")
    
    def add_section(self, title, level=2):
        """Add a section heading"""
        with open(self.report_file, 'a') as f:
            f.write(f"\n{'#' * level} {title}\n\n")
            self.report_content.append(f"\n{'#' * level} {title}\n\n")
    
    def add_text(self, text):
        """Add text content"""
        with open(self.report_file, 'a') as f:
            f.write(f"{text}\n\n")
            self.report_content.append(f"{text}\n\n")
    
    def add_table(self, df, caption=""):
        """Add a table in markdown format"""
        with open(self.report_file, 'a') as f:
            if caption:
                f.write(f"**{caption}**\n\n")
                self.report_content.append(f"**{caption}**\n\n")
            f.write(df.to_markdown())
            f.write("\n\n")
    
    def add_metrics_summary(self, metrics_dict, title="Metrics Summary"):
        """Add metrics as a formatted table"""
        df = pd.DataFrame(metrics_dict, index=[0]).T
        df.columns = ['Value']
        self.add_table(df, caption=title)
    
    def save_and_add_plot(self, fig, filename, caption=""):
        """Save plot and add to report"""
        # Save plot
        plot_path = self.image_folder / f"{filename}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        # Add to report
        with open(self.report_file, 'a') as f:
            if caption:
                f.write(f"**{caption}**\n\n")
                self.report_content.append(f"**{caption}**\n\n")
            f.write(f"![{filename}](images/{filename}.png)\n\n")
            self.report_content.append(f"![{filename}](images/{filename}.png)\n\n")
    
    def finalize_report(self):
        with open(self.report_file, "w") as f:
            f.writelines(self.report_content)
        print(f"\n✓ Report saved to: {self.report_file}")