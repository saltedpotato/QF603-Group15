from vol_models.model_load_packages import *

class VolatilityReportGenerator:
    """
    A unified comprehensive report generator for volatility forecasting analysis.
    Saves plots and outputs in structured markdown format.
    
    Features:
    - Dynamic Table of Contents generation
    - Support for appending to existing reports (multi-run support)
    - Flexible report titles and author attribution
    - Automatic plot and table management
    - Timestamped filenames for versioning
    """
    
    def __init__(self, report_name="volatility_forecast_report", append=False, find_latest=False, 
                 report_title="Volatility Forecasting Report", author="PhD Research Team"):
        """
        Initialize the report generator.
        
        Parameters:
        -----------
        report_name : str
            Base name for the report file (default: "volatility_forecast_report")
        append : bool
            If True, find and append to the latest report file (default: False)
        find_latest : bool
            If True and append=False, still look for latest report to continue (default: False)
        report_title : str
            Title displayed at the top of the report (default: "Volatility Forecasting Report")
        author : str
            Author name to display in report header (default: "PhD Research Team")
        """
        self.report_name = report_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_title = report_title
        self.author = author
        
        # Create directories
        self.base_dir = Path("./report_output_v6")
        self.images_dir = self.base_dir / "images"
        self.base_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        
        # Aliases for backward compatibility
        self.report_folder = self.base_dir
        self.image_folder = self.images_dir
        
        # TOC tracking
        self.toc_entries = []
        self.toc_marker = "<!-- TOC_END -->"
        
        # Try to find and append to latest report
        report_files = sorted(self.base_dir.glob(f"{self.report_name}_*.md"), reverse=True)
        
        if (append or find_latest) and report_files:
            self.report_file = report_files[0]
            print(f"✓ Appending to existing report: {self.report_file}")
        else:
            # Create a new report
            self.report_file = self.base_dir / f"{self.report_name}_{self.timestamp}.md"
            self._init_report()
        
    def _init_report(self):
        """Initialize a new markdown report with dynamic TOC"""
        report_text = f"""# {self.report_title}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Author:** {self.author}

---

## Table of Contents

{self.toc_marker}

---

"""
        with open(self.report_file, 'w') as f:
            f.write(report_text)

    def add_section(self, title, level=2):
        """Add a section heading and update TOC"""
        section_text = f"\n{'#' * level} {title}\n\n"
        with open(self.report_file, 'a') as f:
            f.write(section_text)
        
        # Track this section for TOC
        self._add_to_toc(title, level)
    
    def _add_to_toc(self, title, level):
        """Add entry to TOC (will be written at finalization)"""
        # Create anchor from title (lowercase, replace spaces with hyphens, remove special chars)
        anchor = title.lower()
        anchor = anchor.replace(' ', '-').replace('(', '').replace(')', '').replace('&', '')
        anchor = anchor.replace(',', '').replace(':', '').replace('/', '').replace("'", '')
        
        # Determine indentation based on level
        indent = "    " * (level - 2) if level > 2 else ""
        
        # Create TOC entry
        if level == 2:
            toc_entry = f"{len([e for e in self.toc_entries if not e.startswith(' ')]) + 1}. [{title}](#{anchor})"
        else:
            toc_entry = f"{indent}- [{title}](#{anchor})"
        
        self.toc_entries.append(toc_entry)
    
    def _update_toc_in_file(self):
        """Rewrite the TOC section in the report file (call once at the end)"""
        # Read current content
        with open(self.report_file, 'r') as f:
            content = f.read()
        
        # Find TOC marker position
        if self.toc_marker in content:
            # Split at marker
            before_toc, after_toc = content.split(self.toc_marker, 1)
            
            # Rebuild TOC from scratch
            toc_text = "\n".join(self.toc_entries) + "\n\n"
            
            # Reconstruct content
            new_content = before_toc + toc_text + self.toc_marker + after_toc
            
            # Write back
            with open(self.report_file, 'w') as f:
                f.write(new_content)
        else:
            # Fallback: just append if marker not found
            print("⚠ Warning: TOC marker not found, skipping TOC update")
    
    def add_text(self, text):
        """Add text content"""
        text_block = f"{text}\n\n"
        with open(self.report_file, 'a') as f:
            f.write(text_block)
    
    def add_table(self, df, caption=""):
        """Add a table in markdown format with formatting"""
        # Format floats to 3 decimal places
        df_formatted = df.copy()
        for col in df_formatted.columns:
            if df_formatted[col].dtype in ['float64', 'float32']:
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else x)
        
        # Split table if more than 4 columns
        table_text = ""
        
        if len(df_formatted.columns) <= 4:
            # Table fits - write it directly
            if caption:
                table_text += f"**{caption}**\n\n"
            table_text += df_formatted.to_markdown() + "\n\n"
        else:
            # Split into chunks of 4 columns
            cols = list(df_formatted.columns)
            num_chunks = (len(cols) + 3) // 4  # Ceiling division
            
            for i in range(num_chunks):
                start_idx = i * 4
                end_idx = min((i + 1) * 4, len(cols))
                chunk_cols = cols[start_idx:end_idx]
                df_chunk = df_formatted[chunk_cols]
                
                # Add caption with part number
                if caption:
                    if num_chunks > 1:
                        chunk_caption = f"{caption} (Part {i+1}/{num_chunks})"
                    else:
                        chunk_caption = caption
                    table_text += f"**{chunk_caption}**\n\n"
                
                table_text += df_chunk.to_markdown() + "\n\n"
        
        with open(self.report_file, 'a') as f:
            f.write(table_text)
    
    def add_metrics_summary(self, metrics_dict, title="Metrics Summary"):
        """Add metrics as a formatted table"""
        # Support both dict and table-style formats
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
        code_text = ""
        if title:
            code_text += f"**{title}**\n\n"
        code_text += "```\n"
        code_text += str(output)
        code_text += "\n```\n\n"
        
        with open(self.report_file, 'a') as f:
            f.write(code_text)
    
    def save_and_add_plot(self, fig, filename, caption="", width=800):
        """
        Save matplotlib figure and add to report.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : str
            Base filename (without extension)
        caption : str
            Caption to display with the plot
        width : int
            Unused parameter, kept for backward compatibility
        """
        # Save plot
        plot_path = self.images_dir / f"{filename}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        # Add to report
        plot_text = ""
        if caption:
            plot_text += f"**{caption}**\n\n"
        plot_text += f"![{caption}](images/{filename}.png)\n\n"
        
        with open(self.report_file, 'a') as f:
            f.write(plot_text)
        
        print(f"✓ Saved plot: {filename}.png")
        return str(plot_path)
    
    def finalize_report(self, final_message=None):
        """
        Finalize the report with a closing message and update TOC.
        
        Parameters:
        -----------
        final_message : str, optional
            Custom message to display at end of report
            If None, uses default message
        """
        # First, update the TOC with all collected entries
        self._update_toc_in_file()
        
        # Determine closing message
        if final_message is None:
            final_message = "Report generation completed"
        
        # Then add closing
        closing = f"""
---

**{final_message}**

*Last Updated: {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*
"""
        with open(self.report_file, 'a') as f:
            f.write(closing)
        
        print(f"\n{'='*60}")
        print(f"✓ Report generated successfully!")
        print(f"  Location: {self.report_file}")
        print(f"  Images:   {self.images_dir}")
        print(f"{'='*60}\n")
