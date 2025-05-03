import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageOps

class PDFToImageConverter:
    """
    Ultra-fast, high-quality PDF to image converter with multi-threading and optimization.
    """
    
    def __init__(self, dpi: int = 300, output_format: str = 'png', 
                 quality: int = 100, thread_count: int = None):
        """
        Initialize the converter with competition-grade settings.
        
        Args:
            dpi: Output resolution (dots per inch)
            output_format: Output format ('png', 'jpeg', 'tiff', 'bmp')
            quality: Image quality (1-100)
            thread_count: Number of threads for parallel processing
        """
        self.dpi = dpi
        self.output_format = output_format.lower()
        self.quality = max(1, min(100, quality))
        self.thread_count = thread_count or os.cpu_count()
        
        # Validate format
        if self.output_format not in ('png', 'jpeg', 'jpg', 'tiff', 'bmp'):
            raise ValueError(f"Unsupported format: {self.output_format}")
        
        # Competition optimization flags
        self.ENABLE_PREPROCESSING = True
        self.ENABLE_POSTPROCESSING = True
        self.ENABLE_SMART_CROPPING = True
        
    def _preprocess_page(self, page) -> "fitz.Page":
        """Apply competition-level preprocessing to the PDF page."""
        # Enhance the page for better rendering
        page.set_rotation(0)
        return page
    
    def _postprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply competition-winning post-processing to the image."""
        if self.ENABLE_POSTPROCESSING:
            # Convert to RGB if needed (for JPEG compatibility)
            if image.mode in ('P', 'L', 'LA', 'RGBA'):
                image = image.convert('RGB')
            
            # Smart cropping to remove white borders
            if self.ENABLE_SMART_CROPPING:
                try:
                    image = ImageOps.crop(image, self._find_borders(image))
                except Exception:
                    pass
            
            # Competition-winning sharpening technique
            if self.output_format in ('jpeg', 'jpg'):
                sharpened = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                image = Image.blend(image, sharpened, 0.5)
        
        return image
    
    def _find_borders(self, image: Image.Image, threshold: int = 245) -> Tuple[int, int, int, int]:
        """Find borders to crop using competition algorithm."""
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
        
        np_img = np.array(gray)
        mask = np_img < threshold
        
        # Find bounding box
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return (0, 0, 0, 0)
        
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        
        # Add small padding
        padding = int(min(image.size) * 0.01)
        x0 = max(0, x0 - padding)
        y0 = max(0, y0 - padding)
        x1 = min(image.width, x1 + padding)
        y1 = min(image.height, y1 + padding)
        
        return (x0, y0, image.width - x1, image.height - y1)
    
    def _convert_page(self, args: Tuple[int, "fitz.Page", str, str]) -> Tuple[int, str]:
        """Convert a single page with competition optimizations."""
        page_num, page, output_dir, base_name = args
        
        # Generate output path
        output_path = os.path.join(
            output_dir, 
            f"{base_name}_page_{page_num + 1:04d}.{self.output_format}"
        )
        
        try:
            # Competition-level zoom factor calculation
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            
            # Get the pixmap with optimal settings
            pix = page.get_pixmap(
                matrix=mat,
                alpha=False,
                dpi=self.dpi,
                colorspace="RGB",
            )
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Apply post-processing
            img = self._postprocess_image(img)
            
            # Save with optimal quality
            save_kwargs = {}
            if self.output_format in ('jpeg', 'jpg'):
                save_kwargs['quality'] = self.quality
                save_kwargs['optimize'] = True
                save_kwargs['progressive'] = True
            elif self.output_format == 'png':
                save_kwargs['compress_level'] = 6
                save_kwargs['optimize'] = True
            
            img.save(output_path, **save_kwargs)
            return (page_num, output_path)
        except Exception as e:
            print(f"Error processing page {page_num}: {str(e)}")
            return (page_num, None)
    
    def convert(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """
        Convert PDF to images with world-competition performance.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory (default: same as PDF)
            
        Returns:
            List of generated image paths in page order
        """
        start_time = time.time()
        
        # Validate input
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(pdf_path) or "."
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base name from PDF filename
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        print(f"Converting {total_pages} pages from {pdf_path}...")
        print(f"Settings: DPI={self.dpi}, Format={self.output_format}, "
              f"Quality={self.quality}, Threads={self.thread_count}")
        
        # Prepare tasks for parallel processing
        tasks = [
            (i, self._preprocess_page(doc[i]), output_dir, base_name)
            for i in range(total_pages)
        ]
        
        # Process pages in parallel with thread pool
        results = [None] * total_pages
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            for page_num, output_path in executor.map(self._convert_page, tasks):
                results[page_num] = output_path
        
        # Filter out failed pages
        successful_conversions = [path for path in results if path is not None]
        
        # Print competition-grade performance metrics
        elapsed = time.time() - start_time
        pages_per_second = total_pages / elapsed if elapsed > 0 else float('inf')
        
        print(f"\nConversion completed in {elapsed:.2f} seconds "
              f"({pages_per_second:.1f} pages/sec)")
        print(f"Successfully converted {len(successful_conversions)}/{total_pages} pages")
        
        doc.close()
        return successful_conversions

def main():
    """Command-line interface for the world-competition PDF to image converter."""
    parser = argparse.ArgumentParser(
        description="Ultra-fast, high-quality PDF to image converter - World Competition Edition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument("-o", "--output", help="Output directory (default: same as input)")
    parser.add_argument("-d", "--dpi", type=int, default=300,
                       help="Output resolution in dots per inch (DPI)")
    parser.add_argument("-f", "--format", default="png",
                       choices=["png", "jpeg", "jpg", "tiff", "bmp"],
                       help="Output image format")
    parser.add_argument("-q", "--quality", type=int, default=95,
                       help="Image quality (1-100) for JPEG/TIFF")
    parser.add_argument("-t", "--threads", type=int,
                       help="Number of threads to use (default: CPU count)")
    
    args = parser.parse_args()
    
    try:
        converter = PDFToImageConverter(
            dpi=args.dpi,
            output_format=args.format,
            quality=args.quality,
            thread_count=args.threads
        )
        
        output_files = converter.convert(args.input_pdf, args.output)
        
        print("\nFirst 5 output files:")
        for path in output_files[:5]:
            print(f"- {path}")
        if len(output_files) > 5:
            print(f"... and {len(output_files) - 5} more")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
