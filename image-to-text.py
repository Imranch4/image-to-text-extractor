import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import cv2
import pytesseract
import os
import numpy as np
from PIL import Image, ImageTk
import json
from datetime import datetime
import threading

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\power\Downloads\quiz-game-gh-pages\Image To Text\tesseract.exe'

class TextExtractorApp:
    def __init__(self, root):
        self.root = root
        self.selected_folder = ""
        self.selected_files = []
        self.mode = None
        self.output_dir = "extracted_texts"
        self.setup_ui()
        self.create_output_dir()
        
    def create_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def setup_ui(self):
        """Set up the user interface"""
        self.root.title("Advanced Image to Text Extractor")
        self.root.geometry("900x700")
        
        # Define colors
        self.primary_color = "#2C3E50"
        self.secondary_color = "#ECF0F1"
        self.accent_color = "#1ABC9C"
        self.warning_color = "#E74C3C"
        self.success_color = "#27AE60"
        
        # Apply styles
        style = ttk.Style(self.root)
        style.theme_use("clam")
        
        # Configure styles
        style.configure("TFrame", background=self.primary_color)
        style.configure("TLabel", background=self.primary_color, foreground=self.secondary_color, font=("Helvetica", 12))
        style.configure("TButton", background=self.accent_color, foreground=self.secondary_color, font=("Helvetica", 12, "bold"), borderwidth=0, padding=10)
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Status.TLabel", font=("Helvetica", 10))
        style.map("TButton", background=[("active", "#16A085")])
        
        # Main container
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="📄 Advanced Image to Text Extractor", style="Title.TLabel")
        title_label.pack(pady=(0, 20))
        
        # Selection frame
        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        btn_folder = ttk.Button(selection_frame, text="📁 Select Folder", command=self.select_folder)
        btn_folder.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        btn_files = ttk.Button(selection_frame, text="🖼️ Select Files", command=self.select_files)
        btn_files.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        btn_clear = ttk.Button(selection_frame, text="🗑️ Clear Selection", command=self.clear_selection)
        btn_clear.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        # Selected items label
        self.lbl_selected = ttk.Label(main_frame, text="No folder or files selected", wraplength=800, justify="center")
        self.lbl_selected.pack(pady=10)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="OCR Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # OCR engine selection
        engine_frame = ttk.Frame(settings_frame)
        engine_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(engine_frame, text="OCR Engine:").pack(side=tk.LEFT)
        self.engine_var = tk.StringVar(value="tesseract")
        ttk.Radiobutton(engine_frame, text="Tesseract", variable=self.engine_var, value="tesseract").pack(side=tk.LEFT, padx=10)
        
        # Language selection
        lang_frame = ttk.Frame(settings_frame)
        lang_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(lang_frame, text="Language:").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="eng")
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.lang_var, values=["eng", "fra", "spa", "deu", "ita", "por"], state="readonly", width=10)
        lang_combo.pack(side=tk.LEFT, padx=10)
        
        # Preprocessing options
        preprocess_frame = ttk.Frame(settings_frame)
        preprocess_frame.pack(fill=tk.X, pady=5)
        
        self.deskew_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preprocess_frame, text="Auto-deskew", variable=self.deskew_var).pack(side=tk.LEFT, padx=10)
        
        self.denoise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preprocess_frame, text="Denoise", variable=self.denoise_var).pack(side=tk.LEFT, padx=10)
        
        self.contrast_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preprocess_frame, text="Enhance Contrast", variable=self.contrast_var).pack(side=tk.LEFT, padx=10)
        
        # Output format
        output_frame = ttk.Frame(settings_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output Format:").pack(side=tk.LEFT)
        self.output_format = tk.StringVar(value="txt")
        ttk.Radiobutton(output_frame, text="Text (.txt)", variable=self.output_format, value="txt").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(output_frame, text="JSON (.json)", variable=self.output_format, value="json").pack(side=tk.LEFT, padx=10)
        
        # Process button
        self.btn_process = ttk.Button(main_frame, text="🚀 Process Images", command=self.start_processing)
        self.btn_process.pack(pady=20, fill=tk.X)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", style="Status.TLabel")
        self.status_label.pack(pady=5)
        
        # Preview area
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=10, wrap=tk.WORD)
        self.preview_text.pack(fill=tk.BOTH, expand=True)
        
    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Images")
        if folder:
            self.selected_folder = folder
            self.mode = "folder"
            self.lbl_selected.config(text=f"Folder selected: {folder}\n({self.count_images(folder)} images found)")
            self.update_preview()
    
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Image Files", 
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif")]
        )
        if files:
            self.selected_files = files
            self.mode = "files"
            filenames = [os.path.basename(f) for f in self.selected_files]
            self.lbl_selected.config(text=f"Files selected: {len(filenames)} images")
            self.update_preview()
    
    def clear_selection(self):
        self.selected_folder = ""
        self.selected_files = []
        self.mode = None
        self.lbl_selected.config(text="No folder or files selected")
        self.preview_text.delete(1.0, tk.END)
    
    def count_images(self, folder):
        """Count number of image files in folder"""
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        return len([f for f in os.listdir(folder) if f.lower().endswith(extensions)])
    
    def update_preview(self):
        """Update preview with selected files info"""
        self.preview_text.delete(1.0, tk.END)
        if self.mode == "folder" and self.selected_folder:
            images = [f for f in os.listdir(self.selected_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
            self.preview_text.insert(tk.END, f"Folder: {self.selected_folder}\n")
            self.preview_text.insert(tk.END, f"Images found: {len(images)}\n\n")
            for img in images[:10]:  # Show first 10 files
                self.preview_text.insert(tk.END, f"• {img}\n")
            if len(images) > 10:
                self.preview_text.insert(tk.END, f"... and {len(images) - 10} more\n")
        
        elif self.mode == "files" and self.selected_files:
            self.preview_text.insert(tk.END, f"Selected files: {len(self.selected_files)}\n\n")
            for i, file_path in enumerate(self.selected_files[:10]):
                filename = os.path.basename(file_path)
                self.preview_text.insert(tk.END, f"• {filename}\n")
            if len(self.selected_files) > 10:
                self.preview_text.insert(tk.END, f"... and {len(self.selected_files) - 10} more\n")
    
    def preprocess_image(self, image):
        """Apply preprocessing to improve OCR accuracy"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        if self.denoise_var.get():
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # Enhance contrast
        if self.contrast_var.get():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Deskew
        if self.deskew_var.get():
            gray = self.deskew(gray)
        
        # Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def deskew(self, image):
        """Deskew the image"""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def extract_text_from_image(self, image_path):
        """Extract text from image with preprocessing"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"text": "", "confidence": 0, "error": "Error reading image"}
            
            # Preprocess image
            processed_img = self.preprocess_image(img)
            
            # OCR configuration
            config = f'--oem 3 --psm 6 -l {self.lang_var.get()}'
            
            # Extract text with confidence
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT, config=config)
            
            # Filter text by confidence
            conf_threshold = 30
            text_parts = []
            confidences = []
            
            for i, word in enumerate(data['text']):
                if word.strip() and int(data['conf'][i]) > conf_threshold:
                    text_parts.append(word)
                    confidences.append(int(data['conf'][i]))
            
            extracted_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                "text": extracted_text,
                "confidence": avg_confidence,
                "word_count": len(extracted_text.split()),
                "raw_text": pytesseract.image_to_string(processed_img, config=config)
            }
            
        except Exception as e:
            return {"text": "", "confidence": 0, "error": str(e)}
    
    def start_processing(self):
        """Start processing in a separate thread"""
        if not self.validate_selection():
            return
        
        # Disable process button during processing
        self.btn_process.config(state='disabled')
        self.status_label.config(text="Processing...")
        
        # Start processing in thread
        thread = threading.Thread(target=self.process_images)
        thread.daemon = True
        thread.start()
    
    def validate_selection(self):
        """Validate user selection"""
        images = self.get_image_list()
        if not images:
            messagebox.showerror("Error", "No valid image files found!")
            return False
        return True
    
    def get_image_list(self):
        """Get list of images based on selection mode"""
        images = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        
        if self.mode == "folder" and os.path.isdir(self.selected_folder):
            images = [os.path.join(self.selected_folder, f) for f in os.listdir(self.selected_folder)
                     if f.lower().endswith(valid_extensions)]
        elif self.mode == "files":
            images = [f for f in self.selected_files if f.lower().endswith(valid_extensions)]
        
        return images
    
    def process_images(self):
        """Process all selected images"""
        try:
            images = self.get_image_list()
            total_images = len(images)
            
            if total_images == 0:
                self.show_error("No valid image files found!")
                return
            
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"extracted_text_{timestamp}.{self.output_format.get()}"
            output_path = os.path.join(self.output_dir, output_filename)
            
            results = []
            
            for i, img_path in enumerate(images):
                # Update progress
                progress = (i + 1) / total_images * 100
                self.root.after(0, self.update_progress, progress, f"Processing {i+1}/{total_images}: {os.path.basename(img_path)}")
                
                # Extract text
                result = self.extract_text_from_image(img_path)
                result['filename'] = os.path.basename(img_path)
                result['filepath'] = img_path
                results.append(result)
            
            # Save results
            self.save_results(results, output_path)
            
            # Show completion message
            self.root.after(0, self.processing_complete, results, output_path)
            
        except Exception as e:
            self.root.after(0, self.show_error, f"An error occurred during processing:\n{str(e)}")
    
    def update_progress(self, value, status):
        """Update progress bar and status"""
        self.progress['value'] = value
        self.status_label.config(text=status)
    
    def save_results(self, results, output_path):
        """Save results in selected format"""
        if self.output_format.get() == "json":
            # Save as JSON
            output_data = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_files": len(results),
                    "language": self.lang_var.get(),
                    "total_text_extracted": sum(len(r['text'].split()) for r in results)
                },
                "results": results
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        else:
            # Save as text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Text Extraction Results\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total files processed: {len(results)}\n")
                f.write(f"Language: {self.lang_var.get()}\n")
                f.write("=" * 50 + "\n\n")
                
                for result in results:
                    f.write(f"--- {result['filename']} ---\n")
                    f.write(f"Confidence: {result['confidence']:.1f}%\n")
                    f.write(f"Word count: {result['word_count']}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"{result['text']}\n\n")
    
    def processing_complete(self, results, output_path):
        """Handle processing completion"""
        total_files = len(results)
        successful = len([r for r in results if r['text'].strip()])
        avg_confidence = np.mean([r['confidence'] for r in results if r['text'].strip()])
        
        # Update UI
        self.btn_process.config(state='normal')
        self.progress['value'] = 0
        self.status_label.config(text="Ready")
        
        # Show results in preview
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, f"✅ Processing Complete!\n\n")
        self.preview_text.insert(tk.END, f"• Files processed: {total_files}\n")
        self.preview_text.insert(tk.END, f"• Successful extractions: {successful}\n")
        self.preview_text.insert(tk.END, f"• Average confidence: {avg_confidence:.1f}%\n")
        self.preview_text.insert(tk.END, f"• Output file: {output_path}\n\n")
        
        # Show sample of extracted text
        if successful > 0:
            sample = next((r for r in results if r['text'].strip()), None)
            if sample:
                self.preview_text.insert(tk.END, f"Sample from '{sample['filename']}':\n")
                self.preview_text.insert(tk.END, "-" * 40 + "\n")
                sample_text = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
                self.preview_text.insert(tk.END, f"{sample_text}\n\n")
        
        # Ask user if they want to open the output file
        if messagebox.askyesno("Success", 
                              f"Processing complete!\n\n"
                              f"Files processed: {total_files}\n"
                              f"Successful: {successful}\n"
                              f"Average confidence: {avg_confidence:.1f}%\n\n"
                              f"Output saved to:\n{output_path}\n\n"
                              f"Would you like to open the output file?"):
            os.startfile(output_path)
    
    def show_error(self, message):
        """Show error message and reset UI"""
        messagebox.showerror("Error", message)
        self.btn_process.config(state='normal')
        self.progress['value'] = 0
        self.status_label.config(text="Ready")

def main():
    root = tk.Tk()
    app = TextExtractorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()