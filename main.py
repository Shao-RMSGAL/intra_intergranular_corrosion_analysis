#  import pandas as pd
#  import cv2
#  import numpy as np
#  from io import StringIO
#  import tkinter as tk
#  from tkinter import filedialog, messagebox
#  from PIL import Image, ImageTk

#  # Try to import Cython optimized functions, fall back to pure Python if not available
#  try:
#      from eds_cython import (
#          generate_eds_mask_fast,
#          calculate_region_statistics_fast,
#          create_circle_mask_fast
#      )
#      CYTHON_AVAILABLE = True
#      print("Using Cython optimized functions")
#  except ImportError:
#      CYTHON_AVAILABLE = False
#      print("Cython module not found, using pure Python functions")


#  class EDSAnalyzer:
#      def __init__(self, root):
#          self.root = root
#          self.root.title("EDS Analysis Tool (Cython Optimized)")
#          self.root.geometry("800x900")

#          # Initialize variables
#          self.eds_data = None
#          self.sem_data = None
#          self.eds_original = None  # Store original unscaled EDS data
#          self.sem_original = None  # Store original unscaled SEM data
#          self.max_eds = -1
#          self.mean_eds = -1
#          self.std_eds = -1
#          self.drawing = False
#          self.ix = self.iy = self.fx = self.fy = -1
#          self.eds_copy = None
#          self.sem_mask = None
#          self.eds_mask = None

#          self.setup_gui()

#          # Bind resize event to update image displays
#          self.root.bind('<Configure>', self.on_window_resize)

#      def setup_gui(self):
#          # Create main frame
#          main_frame = tk.Frame(self.root)
#          main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

#          # File selection buttons
#          button_frame = tk.Frame(main_frame)
#          button_frame.pack(fill=tk.X, pady=(0, 10))

#          self.eds_button = tk.Button(button_frame, text="Select EDS CSV",
#                                      command=self.select_eds_file, width=15)
#          self.eds_button.pack(side=tk.LEFT, padx=(0, 10))

#          self.sem_button = tk.Button(button_frame, text="Select SEM Image",
#                                      command=self.select_sem_file, width=15)
#          self.sem_button.pack(side=tk.LEFT, padx=(0, 10))

#          self.bulk_button = tk.Button(button_frame, text="Select Bulk Cr Content",
#                                       command=self.select_bulk, width=20, state=tk.DISABLED)
#          self.bulk_button.pack(side=tk.LEFT)

#          # File path labels
#          path_frame = tk.Frame(main_frame)
#          path_frame.pack(fill=tk.X, pady=(0, 10))

#          tk.Label(path_frame, text="EDS File:").grid(
#              row=0, column=0, sticky="w")
#          self.eds_path_label = tk.Label(path_frame, text="No file selected",
#                                         fg="gray", wraplength=400)
#          self.eds_path_label.grid(row=0, column=1, sticky="w", padx=(10, 0))

#          tk.Label(path_frame, text="SEM File:").grid(
#              row=1, column=0, sticky="w")
#          self.sem_path_label = tk.Label(path_frame, text="No file selected",
#                                         fg="gray", wraplength=400)
#          self.sem_path_label.grid(row=1, column=1, sticky="w", padx=(10, 0))

#          # Optimization status label
#          opt_frame = tk.Frame(main_frame)
#          opt_frame.pack(fill=tk.X, pady=(0, 10))

#          opt_status = "Cython Optimized" if CYTHON_AVAILABLE else "Pure Python (Install Cython for better performance)"
#          opt_color = "green" if CYTHON_AVAILABLE else "orange"
#          tk.Label(opt_frame, text=f"Status: {opt_status}", fg=opt_color, font=(
#              "Arial", 9)).pack()

#          # Image display frame
#          image_frame = tk.Frame(main_frame)
#          image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

#          # EDS image display
#          eds_frame = tk.LabelFrame(image_frame, text="EDS Image")
#          eds_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

#          self.eds_label = tk.Label(eds_frame, text="No EDS image loaded",
#                                    bg="lightgray", width=30, height=15)
#          self.eds_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

#          # SEM image display
#          sem_frame = tk.LabelFrame(image_frame, text="SEM Image")
#          sem_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

#          self.sem_label = tk.Label(sem_frame, text="No SEM image loaded",
#                                    bg="lightgray", width=30, height=15)
#          self.sem_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

#          # Statistics frame
#          stats_frame = tk.LabelFrame(
#              main_frame, text="Bulk Chromium Content Statistics")
#          stats_frame.pack(fill=tk.X, pady=(10, 0))

#          stats_grid = tk.Frame(stats_frame)
#          stats_grid.pack(padx=10, pady=10)

#          tk.Label(stats_grid, text="Mean:").grid(row=0, column=0, sticky="w")
#          self.mean_label = tk.Label(
#              stats_grid, text="Not calculated", fg="gray")
#          self.mean_label.grid(row=0, column=1, sticky="w", padx=(10, 20))

#          tk.Label(stats_grid, text="Standard Deviation:").grid(
#              row=0, column=2, sticky="w")
#          self.std_label = tk.Label(stats_grid, text="Not calculated", fg="gray")
#          self.std_label.grid(row=0, column=3, sticky="w", padx=(10, 0))

#          # Threshold controls frame
#          threshold_frame = tk.LabelFrame(main_frame, text="Threshold Controls")
#          threshold_frame.pack(fill=tk.X, pady=(10, 0))

#          # SEM threshold slider
#          sem_thresh_frame = tk.Frame(threshold_frame)
#          sem_thresh_frame.pack(fill=tk.X, padx=10, pady=5)

#          tk.Label(sem_thresh_frame, text="SEM Threshold:").pack(side=tk.LEFT)
#          self.sem_threshold_var = tk.IntVar(value=0)
#          self.sem_threshold_slider = tk.Scale(sem_thresh_frame, from_=0, to=255,
#                                               orient=tk.HORIZONTAL, variable=self.sem_threshold_var,
#                                               command=self.update_masks,
#                                               length=300)
#          self.sem_threshold_slider.pack(side=tk.LEFT, padx=(10, 0))

#          # EDS percentage threshold entry
#          eds_thresh_frame = tk.Frame(threshold_frame)
#          eds_thresh_frame.pack(fill=tk.X, padx=10, pady=5)

#          tk.Label(eds_thresh_frame, text="EDS Percentage Threshold:").pack(
#              side=tk.LEFT)
#          self.eds_threshold_var = tk.StringVar(value="5")
#          self.eds_threshold_entry = tk.Entry(
#              eds_thresh_frame, textvariable=self.eds_threshold_var, width=10)
#          self.eds_threshold_entry.pack(side=tk.LEFT, padx=(10, 0))

#          tk.Button(eds_thresh_frame, text="Update Masks",
#                    command=self.update_masks).pack(side=tk.LEFT, padx=(10, 0))

#      def select_eds_file(self):
#          filepath = filedialog.askopenfilename(
#              title="Select EDS CSV file",
#              filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
#          )

#          if filepath:
#              try:
#                  self.eds_original = self.load_csv(filepath)
#                  self.eds_data = self.eds_original.copy()  # Working copy for display
#                  self.eds_path_label.config(text=filepath, fg="black")
#                  self.display_image(self.eds_data, self.eds_label, "EDS")
#                  self.check_enable_bulk_button()

#                  # If SEM is already loaded, rescale it to match EDS
#                  if self.sem_original is not None:
#                      self.rescale_sem_to_eds()

#              except Exception as e:
#                  messagebox.showerror(
#                      "Error", f"Failed to load EDS file: {str(e)}")

#      def select_sem_file(self):
#          filepath = filedialog.askopenfilename(
#              title="Select SEM TIFF file",
#              filetypes=[("TIFF files", "*.tiff"),
#                         ("TIFF files", "*.tif"), ("All files", "*.*")]
#          )

#          if filepath:
#              try:
#                  sem_temp = cv2.imread(filepath)
#                  if sem_temp is None:
#                      raise ValueError("Could not load image file")

#                  # Trim 21 pixels from top and bottom
#                  height = sem_temp.shape[0]
#                  self.sem_original = sem_temp[21:height-21, :]

#                  # Scale to match EDS dimensions if EDS is loaded
#                  if self.eds_original is not None:
#                      self.rescale_sem_to_eds()
#                  else:
#                      self.sem_data = self.sem_original.copy()

#                  self.sem_path_label.config(text=filepath, fg="black")
#                  self.display_image(self.sem_data, self.sem_label, "SEM")
#              except Exception as e:
#                  messagebox.showerror(
#                      "Error", f"Failed to load SEM file: {str(e)}")

#      def load_csv(self, filepath):
#          # Clean training commas
#          with open(filepath, "r") as f:
#              lines = [line.rstrip(',\n') + '\n' for line in f]

#          # Read CSV without headers
#          eds_df = pd.read_csv(StringIO(''.join(lines)), header=None)
#          matrix = eds_df.values

#          # Check shape of the matrix
#          height, width = np.shape(matrix)
#          self.max_eds = matrix.max()

#          if height != width:
#              raise ValueError(
#                  f"Incorrect dimensions of matrix. Must be square. Shape is ({height}, {width})")

#          # Convert to float first to handle any potential NaN values
#          if matrix.min() < 0:
#              raise ValueError("Matrix contains negative values.")

#          # Handle NaN values (replace with 0 or interpolate)
#          if np.isnan(matrix).any():
#              raise ValueError("Matrix contains nan values.")

#          matrix = (matrix * 255 / matrix.max()).astype(np.uint8)
#          return cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

#      def display_image(self, cv_image, label, image_type):
#          # Convert BGR to RGB for PIL
#          if len(cv_image.shape) == 3:
#              rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
#          else:
#              rgb_image = cv_image

#          # Convert to PIL Image
#          pil_image = Image.fromarray(rgb_image)

#          # Get actual label dimensions
#          label.update_idletasks()  # Ensure label has been drawn
#          label_width = label.winfo_width()
#          label_height = label.winfo_height()

#          # Use minimum size if label hasn't been rendered yet
#          if label_width <= 1 or label_height <= 1:
#              label_width = 300
#              label_height = 250

#          # Calculate size maintaining aspect ratio to fill available space
#          img_width, img_height = pil_image.size
#          ratio = min(label_width/img_width, label_height/img_height)
#          new_width = int(img_width * ratio)
#          new_height = int(img_height * ratio)

#          pil_image = pil_image.resize(
#              (new_width, new_height), Image.Resampling.LANCZOS)

#          # Convert to PhotoImage and display
#          photo = ImageTk.PhotoImage(pil_image)
#          label.config(image=photo, text="")
#          label.image = photo  # Keep a reference

#      def check_enable_bulk_button(self):
#          if self.eds_original is not None:
#              self.bulk_button.config(state=tk.NORMAL)

#      def on_window_resize(self, event):
#          # Only update images when the main window is resized, not child widgets
#          if event.widget == self.root:
#              # Use after_idle to prevent multiple rapid updates during resize
#              self.root.after_idle(self.update_image_displays)

#      def update_image_displays(self):
#          # Update EDS image display if loaded
#          if self.eds_data is not None:
#              self.display_image(self.eds_data, self.eds_label, "EDS")

#          # Update SEM image display if loaded
#          if self.sem_data is not None:
#              self.display_image(self.sem_data, self.sem_label, "SEM")

#      def rescale_sem_to_eds(self):
#          """Scale SEM image to match EDS dimensions"""
#          if self.eds_original is not None and self.sem_original is not None:
#              eds_height, eds_width = self.eds_original.shape[:2]
#              self.sem_data = cv2.resize(
#                  self.sem_original, (eds_width, eds_height))
#              if hasattr(self, 'sem_label'):
#                  self.display_image(self.sem_data, self.sem_label, "SEM")

#      def update_masks(self, event=None):
#          """Update both SEM and EDS masks based on threshold values"""
#          if self.sem_data is None or self.eds_data is None:
#              return

#          try:
#              # Get threshold values
#              sem_threshold = self.sem_threshold_var.get()
#              eds_threshold_percent = float(self.eds_threshold_var.get())

#              # Generate SEM mask (pixels below threshold)
#              sem_gray = cv2.cvtColor(self.sem_data, cv2.COLOR_BGR2GRAY)
#              self.sem_mask = sem_gray < sem_threshold

#              # Generate EDS mask based on SEM mask
#              self.generate_eds_mask(eds_threshold_percent)

#              # Apply masks and update displays
#              self.apply_masks()
#          except ValueError:
#              messagebox.showerror("Error", "Invalid threshold percentage value")

#      def generate_eds_mask(self, threshold_percent):
#          """Generate EDS mask by expanding circles around SEM mask pixels"""
#          if CYTHON_AVAILABLE:
#              # Use Cython optimized version
#              eds_gray = cv2.cvtColor(self.eds_original, cv2.COLOR_BGR2GRAY)
#              sem_mask_uint8 = self.sem_mask.astype(np.uint8) * 255

#              self.eds_mask = generate_eds_mask_fast(
#                  eds_gray, sem_mask_uint8, self.max_eds, threshold_percent
#              ).astype(bool)
#          else:
#              # Fall back to original Python implementation
#              self._generate_eds_mask_python(threshold_percent)

#      def _generate_eds_mask_python(self, threshold_percent):
#          """Original Python implementation for fallback"""
#          self.eds_mask = np.zeros(self.eds_original.shape[:2], dtype=bool)

#          # Convert EDS to grayscale for calculations
#          eds_gray = cv2.cvtColor(self.eds_original, cv2.COLOR_BGR2GRAY)

#          # Calculate threshold value as percentage of max EDS value
#          threshold_value = (threshold_percent / 100.0) * \
#              self.max_eds * (255.0 / self.max_eds)

#          # Find all pixels in SEM mask
#          sem_mask_pixels = np.where(self.sem_mask)

#          for y, x in zip(sem_mask_pixels[0], sem_mask_pixels[1]):
#              # Start with radius 1 and expand until condition is met
#              radius = 1
#              max_radius = min(eds_gray.shape) // 4  # Prevent infinite loops

#              while radius <= max_radius:
#                  # Create circular mask
#                  circle_mask = self.create_circle_mask(
#                      eds_gray.shape, x, y, radius)

#                  # Calculate average value in circle
#                  if np.any(circle_mask):
#                      avg_value = np.mean(eds_gray[circle_mask])

#                      # If average is >= threshold, stop expanding
#                      if avg_value >= threshold_value:
#                          break

#                  radius += 1

#              # Add final circle to EDS mask
#              final_circle = self.create_circle_mask(
#                  eds_gray.shape, x, y, radius)
#              self.eds_mask |= final_circle

#      def create_circle_mask(self, shape, center_x, center_y, radius):
#          """Create a circular mask"""
#          if CYTHON_AVAILABLE:
#              # Use Cython optimized version
#              return create_circle_mask_fast(
#                  shape[0], shape[1], center_x, center_y, radius
#              ).astype(bool)
#          else:
#              # Fall back to original NumPy implementation
#              y, x = np.ogrid[:shape[0], :shape[1]]
#              mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
#              return mask

#      def apply_masks(self):
#          """Apply masks as overlays to the images"""
#          # Apply red overlay to SEM image
#          sem_display = self.sem_data.copy()
#          if self.sem_mask is not None:
#              sem_display[self.sem_mask] = [0, 0, 255]  # Red overlay

#          # Apply green overlay to EDS image
#          eds_display = self.eds_data.copy()
#          if self.eds_mask is not None:
#              eds_display[self.eds_mask] = [0, 255, 0]  # Green overlay

#          # Update displays
#          self.display_image(sem_display, self.sem_label, "SEM")
#          self.display_image(eds_display, self.eds_label, "EDS")

#      def select_bulk(self):
#          if self.eds_original is None:
#              messagebox.showerror("Error", "No EDS image loaded")
#              return

#          # Create OpenCV window for selection using original EDS data
#          self.eds_copy = self.eds_original.copy()
#          window_name = 'Click and drag to select bulk chromium content area'

#          cv2.namedWindow(window_name)
#          cv2.setMouseCallback(window_name, self.draw_rectangle)
#          cv2.imshow(window_name, self.eds_original)

#          messagebox.showinfo("Instructions",
#                              "Click and drag to select a rectangular region that represents the bulk chromium content.\n"
#                              "Press 'q' to finish selection.")

#          while True:
#              key = cv2.waitKey(1) & 0xFF
#              if key == ord('q'):
#                  break

#          cv2.destroyAllWindows()

#      def draw_rectangle(self, event, x, y, flags, param):
#          if event == cv2.EVENT_LBUTTONDOWN:
#              self.drawing = True
#              self.ix, self.iy = x, y
#              self.eds_copy = self.eds_original.copy()

#          elif event == cv2.EVENT_MOUSEMOVE:
#              if self.drawing:
#                  eds_display = self.eds_copy.copy()
#                  cv2.rectangle(eds_display, (self.ix, self.iy),
#                                (x, y), (0, 255, 0), 1)
#                  cv2.imshow(
#                      'Click and drag to select bulk chromium content area', eds_display)

#          elif event == cv2.EVENT_LBUTTONUP:
#              self.drawing = False
#              self.fx, self.fy = x, y
#              if self.ix == self.iy or self.fx == self.fy:
#                  return
#              cv2.rectangle(self.eds_copy, (self.ix, self.iy),
#                            (self.fx, self.fy), (0, 255, 0), 1)
#              cv2.imshow(
#                  'Click and drag to select bulk chromium content area', self.eds_copy)

#              # Calculate statistics for the selected region
#              self.calculate_statistics(self.ix, self.iy, self.fx, self.fy)

#      def calculate_statistics(self, x1, y1, x2, y2):
#          # Ensure coordinates are ordered correctly
#          x_start, x_end = sorted([x1, x2])
#          y_start, y_end = sorted([y1, y2])

#          # Extract the region of interest from original EDS data
#          roi = self.eds_original[y_start:y_end, x_start:x_end]

#          if CYTHON_AVAILABLE:
#              # Use Cython optimized statistics calculation
#              self.mean_eds, self.std_eds = calculate_region_statistics_fast(
#                  roi, self.max_eds
#              )
#          else:
#              # Fall back to original NumPy implementation
#              self.mean_eds = np.mean(roi) / 255 * self.max_eds
#              self.std_eds = np.std(roi) / 255 * self.max_eds

#          # Update GUI labels
#          self.mean_label.config(text=f"{self.mean_eds:.2f}", fg="black")
#          self.std_label.config(text=f"{self.std_eds:.2f}", fg="black")

#          print(f"\nSelected Region Statistics:")
#          print(f"Region size: {roi.shape[1]}x{roi.shape[0]} pixels")
#          print(f"Average pixel value: {self.mean_eds:.2f}")
#          print(f"Standard deviation: {self.std_eds:.2f}")


#  def main():
#      root = tk.Tk()
#      app = EDSAnalyzer(root)
#      root.mainloop()


#  if __name__ == "__main__":
#      main()

import pandas as pd
import cv2
import numpy as np
from io import StringIO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Try to import Cython optimized functions,
# fall back to pure Python if not available
try:
    from eds_cython import (
        generate_eds_mask_fast,
        calculate_region_statistics_fast,
        create_circle_mask_fast
    )
    CYTHON_AVAILABLE = True
    print("Using Cython optimized functions")
except ImportError:
    CYTHON_AVAILABLE = False
    print("Cython module not found, using pure Python functions")


class ToolTip:
    """Simple tooltip implementation"""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, _event=None):
        if self.tooltip:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip, text=self.text,
                         background="lightyellow",
                         relief="solid",
                         borderwidth=1,
                         font=("Arial", 9),
                         wraplength=300)
        label.pack()

    def hide_tooltip(self, _event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class EDSAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("EDS Analysis Tool (Cython Optimized)")
        self.root.geometry("800x950")

        # Initialize variables
        self.eds_data = np.zeros((0, 0), dtype=np.uint8)
        self.sem_data = np.zeros((0, 0), dtype=np.uint8)
        # Store original unscaled EDS data
        self.eds_original = np.zeros((0, 0), dtype=np.uint8)
        # Store original unscaled SEM data
        self.sem_original = np.zeros((0, 0), dtype=np.uint8)
        self.max_eds = -1
        self.mean_eds = -1
        self.std_eds = -1
        self.drawing = False
        self.ix = self.iy = self.fx = self.fy = -1
        self.eds_copy = np.zeros((0, 0), dtype=np.uint8)
        self.sem_mask = np.zeros((0, 0), dtype=np.uint8)
        self.eds_mask = np.zeros((0, 0), dtype=np.uint8)
        # New: mask for excluded areas
        self.exclusion_mask = np.zeros((0, 0), dtype=np.uint8)
        self.bulk_selected = False  # Track if bulk Cr content was selected
        self.window_name = ""
        self.ex_window_name = ""

        # Variables for exclusion area selection
        self.exclusion_rects = []  # List to store exclusion rectangles
        self.current_exclusion_rect = np.zeros((0, 0), dtype=np.uint8)

        self.setup_gui()

        # Bind resize event to update image displays
        self.root.bind('<Configure>', self.on_window_resize)

    def setup_gui(self):
        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # File selection buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.eds_button = tk.Button(button_frame, text="Select EDS CSV",
                                    command=self.select_eds_file, width=15)
        self.eds_button.pack(side=tk.LEFT, padx=(0, 10))

        self.sem_button = tk.Button(button_frame, text="Select SEM Image",
                                    command=self.select_sem_file, width=15)
        self.sem_button.pack(side=tk.LEFT, padx=(0, 10))

        self.bulk_button = tk.Button(button_frame,
                                     text="Select Bulk Cr Content",
                                     command=self.select_bulk,
                                     width=20,
                                     state=tk.DISABLED)
        self.bulk_button.pack(side=tk.LEFT, padx=(0, 10))

        # New button for exclusion areas
        self.exclusion_button = tk.Button(button_frame,
                                          text="Select Excluded Areas",
                                          command=self.select_exclusion_areas,
                                          width=20,
                                          state=tk.DISABLED)
        self.exclusion_button.pack(side=tk.LEFT)

        # File path labels
        path_frame = tk.Frame(main_frame)
        path_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(path_frame, text="EDS File:").grid(
            row=0, column=0, sticky="w")
        self.eds_path_label = tk.Label(path_frame, text="No file selected",
                                       fg="gray", wraplength=400)
        self.eds_path_label.grid(row=0, column=1, sticky="w", padx=(10, 0))

        tk.Label(path_frame, text="SEM File:").grid(
            row=1, column=0, sticky="w")
        self.sem_path_label = tk.Label(path_frame, text="No file selected",
                                       fg="gray", wraplength=400)
        self.sem_path_label.grid(row=1, column=1, sticky="w", padx=(10, 0))

        # Optimization status label
        opt_frame = tk.Frame(main_frame)
        opt_frame.pack(fill=tk.X, pady=(0, 10))

        opt_status = "Cython Optimized" if CYTHON_AVAILABLE else \
            "Pure Python (Install Cython for better performance)"
        opt_color = "green" if CYTHON_AVAILABLE else "orange"
        tk.Label(opt_frame, text=f"Status: {opt_status}", fg=opt_color, font=(
            "Arial", 9)).pack()

        # Image display frame
        image_frame = tk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # EDS image display
        eds_frame = tk.LabelFrame(image_frame, text="EDS Image")
        eds_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.eds_label = tk.Label(eds_frame, text="No EDS image loaded",
                                  bg="lightgray", width=30, height=15)
        self.eds_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # SEM image display
        sem_frame = tk.LabelFrame(image_frame, text="SEM Image")
        sem_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.sem_label = tk.Label(sem_frame, text="No SEM image loaded",
                                  bg="lightgray", width=30, height=15)
        self.sem_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Statistics frame
        stats_frame = tk.LabelFrame(
            main_frame, text="Bulk Chromium Content Statistics")
        stats_frame.pack(fill=tk.X, pady=(10, 0))

        stats_grid = tk.Frame(stats_frame)
        stats_grid.pack(padx=10, pady=10)

        tk.Label(stats_grid, text="Mean:").grid(row=0, column=0, sticky="w")
        self.mean_label = tk.Label(
            stats_grid, text="Not calculated", fg="gray")
        self.mean_label.grid(row=0, column=1, sticky="w", padx=(10, 20))

        tk.Label(stats_grid, text="Standard Deviation:").grid(
            row=0, column=2, sticky="w")
        self.std_label = tk.Label(stats_grid, text="Not calculated", fg="gray")
        self.std_label.grid(row=0, column=3, sticky="w", padx=(10, 0))

        # Threshold controls frame
        threshold_frame = tk.LabelFrame(main_frame, text="Threshold Controls")
        threshold_frame.pack(fill=tk.X, pady=(10, 0))

        # SEM threshold slider
        sem_thresh_frame = tk.Frame(threshold_frame)
        sem_thresh_frame.pack(fill=tk.X, padx=10, pady=5)

        sem_label = tk.Label(sem_thresh_frame, text="SEM Threshold:")
        sem_label.pack(side=tk.LEFT)
        self.sem_threshold_var = tk.IntVar(value=0)
        self.sem_threshold_slider = tk.Scale(sem_thresh_frame, from_=0, to=255,
                                             orient=tk.HORIZONTAL,
                                             variable=self.sem_threshold_var,
                                             command=self.update_masks,
                                             length=300, state=tk.DISABLED)
        self.sem_threshold_slider.pack(side=tk.LEFT, padx=(10, 0))

        # Add tooltip for SEM threshold
        ToolTip(sem_label,
                "Select Bulk Cr Content first to enable threshold controls")
        ToolTip(self.sem_threshold_slider,
                "Select Bulk Cr Content first to enable threshold controls")

        # EDS percentage threshold entry
        eds_thresh_frame = tk.Frame(threshold_frame)
        eds_thresh_frame.pack(fill=tk.X, padx=10, pady=5)

        eds_label = tk.Label(
            eds_thresh_frame, text="EDS Percentage Threshold:")
        eds_label.pack(side=tk.LEFT)
        self.eds_threshold_var = tk.StringVar(value="5")
        self.eds_threshold_entry = tk.Entry(
            eds_thresh_frame, textvariable=self.eds_threshold_var,
            width=10, state=tk.DISABLED)
        self.eds_threshold_entry.pack(side=tk.LEFT, padx=(10, 0))

        self.update_masks_button = tk.Button(eds_thresh_frame,
                                             text="Update Masks",
                                             command=self.update_masks,
                                             state=tk.DISABLED)
        self.update_masks_button.pack(side=tk.LEFT, padx=(10, 0))

        # Add tooltips for EDS threshold controls
        ToolTip(eds_label,
                "Select Bulk Cr Content first to enable threshold controls")
        ToolTip(self.eds_threshold_entry,
                "Select Bulk Cr Content first to enable threshold controls")
        ToolTip(self.update_masks_button,
                "Select Bulk Cr Content first to enable threshold controls")

        # Exclusion areas info
        exclusion_info_frame = tk.LabelFrame(
            main_frame, text="Exclusion Areas")
        exclusion_info_frame.pack(fill=tk.X, pady=(10, 0))

        self.exclusion_info_label = tk.Label(
            exclusion_info_frame,
            text="No exclusion areas selected",
            fg="gray")
        self.exclusion_info_label.pack(padx=10, pady=5)

        clear_exclusions_button = tk.Button(exclusion_info_frame,
                                            text="Clear All Exclusions",
                                            command=self.clear_exclusions,
                                            state=tk.DISABLED)
        clear_exclusions_button.pack(pady=5)
        self.clear_exclusions_button = clear_exclusions_button

    def select_eds_file(self):
        filepath = filedialog.askopenfilename(
            title="Select EDS CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filepath:
            try:
                self.eds_original = self.load_csv(filepath)
                self.eds_data = self.eds_original.copy()  # Display copy
                self.eds_path_label.config(text=filepath, fg="black")
                self.display_image(self.eds_data, self.eds_label, "EDS")
                self.check_enable_bulk_button()

                # If SEM is already loaded, rescale it to match EDS
                if self.sem_original.shape != (0, 0):
                    self.rescale_sem_to_eds()

            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to load EDS file: {str(e)}")

    def select_sem_file(self):
        filepath = filedialog.askopenfilename(
            title="Select SEM TIFF file",
            filetypes=[("TIFF files", "*.tiff"),
                       ("TIFF files", "*.tif"), ("All files", "*.*")]
        )

        if filepath:
            try:
                sem_temp = cv2.imread(filepath)
                if sem_temp.shape == (0, 0):
                    raise ValueError("Could not load image file")

                # Trim 21 pixels from top and bottom
                height = sem_temp.shape[0]
                self.sem_original = sem_temp[21:height-21, :]

                # Scale to match EDS dimensions if EDS is loaded
                if self.eds_original.shape != (0, 0):
                    self.rescale_sem_to_eds()
                else:
                    self.sem_data = self.sem_original.copy()

                self.sem_path_label.config(text=filepath, fg="black")
                self.display_image(self.sem_data, self.sem_label, "SEM")
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to load SEM file: {str(e)}")

    def load_csv(self, filepath):
        # Clean training commas
        with open(filepath, "r") as f:
            lines = [line.rstrip(',\n') + '\n' for line in f]

        # Read CSV without headers
        eds_df = pd.read_csv(StringIO(''.join(lines)), header=None)
        matrix = eds_df.values

        # Check shape of the matrix
        height, width = np.shape(matrix)
        self.max_eds = matrix.max()

        if height != width:
            raise ValueError(
                f"Incorrect dimensions of matrix. Must be square. \
                        Shape is ({height}, {width})")

        # Convert to float first to handle any potential NaN values
        if matrix.min() < 0:
            raise ValueError("Matrix contains negative values.")

        # Handle NaN values (replace with 0 or interpolate)
        if np.isnan(matrix).any():
            raise ValueError("Matrix contains nan values.")

        matrix = (matrix * 255 / matrix.max()).astype(np.uint8)
        return cv2.cvtColor(matrix, cv2.COLOR_GRAY2BGR)

    def display_image(self, cv_image, label, _):
        # Convert BGR to RGB for PIL
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv_image

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)

        # Get actual label dimensions
        label.update_idletasks()  # Ensure label has been drawn
        label_width = label.winfo_width()
        label_height = label.winfo_height()

        # Use minimum size if label hasn't been rendered yet
        if label_width <= 1 or label_height <= 1:
            label_width = 300
            label_height = 250

        # Calculate size maintaining aspect ratio to fill available space
        img_width, img_height = pil_image.size
        ratio = min(label_width/img_width, label_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)

        pil_image = pil_image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(pil_image)
        label.config(image=photo, text="")
        label.image = photo  # Keep a reference

    def check_enable_bulk_button(self):
        if self.eds_original.shape != (0, 0):
            self.bulk_button.config(state=tk.NORMAL)

    def enable_threshold_controls(self):
        """Enable threshold controls after bulk Cr selection"""
        self.sem_threshold_slider.config(state=tk.NORMAL)
        self.eds_threshold_entry.config(state=tk.NORMAL)
        self.update_masks_button.config(state=tk.NORMAL)
        self.exclusion_button.config(state=tk.NORMAL)

    def on_window_resize(self, event):
        # Only update images when the main window is resized, not child widgets
        if event.widget == self.root:
            # Use after_idle to prevent multiple rapid updates during resize
            self.root.after_idle(self.update_image_displays)

    def update_image_displays(self):
        # Update EDS image display if loaded
        if self.eds_data.shape != (0, 0):
            self.display_image(self.eds_data, self.eds_label, "EDS")

        # Update SEM image display if loaded
        if self.sem_data.shape != (0, 0):
            self.display_image(self.sem_data, self.sem_label, "SEM")

    def rescale_sem_to_eds(self):
        """Scale SEM image to match EDS dimensions"""
        if self.eds_original.shape != (0, 0) \
                and self.sem_original.shape != (0, 0):
            eds_height, eds_width = self.eds_original.shape[:2]
            self.sem_data = cv2.resize(
                self.sem_original, (eds_width, eds_height))
            if hasattr(self, 'sem_label'):
                self.display_image(self.sem_data, self.sem_label, "SEM")

    def update_masks(self, event=None):
        """Update both SEM and EDS masks based on threshold values"""
        if self.sem_data.shape == (0, 0) or self.eds_data.shape == (0, 0) \
                or not self.bulk_selected:
            return

        try:
            # Get threshold values
            sem_threshold = self.sem_threshold_var.get()
            eds_threshold_percent = float(self.eds_threshold_var.get())

            # Generate SEM mask (pixels below threshold)
            sem_gray = cv2.cvtColor(self.sem_data, cv2.COLOR_BGR2GRAY)
            self.sem_mask = sem_gray < sem_threshold

            # Apply exclusion mask if it exists
            if self.exclusion_mask.shape != (0, 0):
                self.sem_mask = self.sem_mask & ~self.exclusion_mask

            # Generate EDS mask based on SEM mask
            self.generate_eds_mask(eds_threshold_percent)

            # Apply masks and update displays
            self.apply_masks()
        except ValueError as e:
            messagebox.showerror("Error", "Invalid threshold percentage value")
            print(e)

    def generate_eds_mask(self, threshold_percent):
        """Generate EDS mask by expanding circles around SEM mask pixels"""
        if CYTHON_AVAILABLE:
            # Use Cython optimized version
            eds_gray = cv2.cvtColor(self.eds_original, cv2.COLOR_BGR2GRAY)
            sem_mask_uint8 = self.sem_mask.astype(np.uint8) * 255

            self.eds_mask = generate_eds_mask_fast(
                eds_gray, sem_mask_uint8, self.max_eds, threshold_percent
            ).astype(bool)
        else:
            # Fall back to original Python implementation
            self._generate_eds_mask_python(threshold_percent)

    def _generate_eds_mask_python(self, threshold_percent):
        """Original Python implementation for fallback"""
        self.eds_mask = np.zeros(self.eds_original.shape[:2], dtype=bool)

        # Convert EDS to grayscale for calculations
        eds_gray = cv2.cvtColor(self.eds_original, cv2.COLOR_BGR2GRAY)

        # Calculate threshold value as percentage of max EDS value
        threshold_value = (threshold_percent / 100.0) * \
            self.max_eds * (255.0 / self.max_eds)

        # Find all pixels in SEM mask
        sem_mask_pixels = np.where(self.sem_mask)

        for y, x in zip(sem_mask_pixels[0], sem_mask_pixels[1]):
            # Start with radius 1 and expand until condition is met
            radius = 1
            max_radius = min(eds_gray.shape) // 4  # Prevent infinite loops

            while radius <= max_radius:
                # Create circular mask
                circle_mask = self.create_circle_mask(
                    eds_gray.shape, x, y, radius)

                # Calculate average value in circle
                if np.any(circle_mask):
                    avg_value = np.mean(eds_gray[circle_mask])

                    # If average is >= threshold, stop expanding
                    if avg_value >= threshold_value:
                        break

                radius += 1

            # Add final circle to EDS mask
            final_circle = self.create_circle_mask(
                eds_gray.shape, x, y, radius)
            self.eds_mask |= final_circle

    def create_circle_mask(self, shape, center_x, center_y, radius):
        """Create a circular mask"""
        if CYTHON_AVAILABLE:
            # Use Cython optimized version
            return create_circle_mask_fast(
                shape[0], shape[1], center_x, center_y, radius
            ).astype(bool)
        else:
            # Fall back to original NumPy implementation
            y, x = np.ogrid[:shape[0], :shape[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            return mask

    def apply_masks(self):
        """Apply masks as overlays to the images"""
        # Apply red overlay to SEM image for threshold areas
        sem_display = self.sem_data.copy()
        if self.sem_mask.shape != (0, 0):
            sem_display[self.sem_mask] = [0, 0, 255]  # Red overlay

        # Apply blue overlay for exclusion areas
        if self.exclusion_mask.shape != (0, 0):
            sem_display[self.exclusion_mask] = [255, 0, 0]  # Blue overlay

        # Apply green overlay to EDS image
        eds_display = self.eds_data.copy()
        if self.eds_mask.shape != (0, 0):
            eds_display[self.eds_mask] = [0, 255, 0]  # Green overlay

        # Update displays
        self.display_image(sem_display, self.sem_label, "SEM")
        self.display_image(eds_display, self.eds_label, "EDS")

    def select_bulk(self):
        if self.eds_original.shape == (0, 0):
            messagebox.showerror("Error", "No EDS image loaded")
            return

        # Create OpenCV window for selection using original EDS data
        self.eds_copy = self.eds_original.copy()
        self.window_name = 'Click and drag to select bulk chromium content area. \
Press \'q\' to quit.'

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.draw_rectangle_bulk)
        cv2.imshow(self.window_name, self.eds_original)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or not \
                    cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE):
                break

        cv2.destroyAllWindows()

    def draw_rectangle_bulk(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.eds_copy = self.eds_original.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                eds_display = self.eds_copy.copy()
                cv2.rectangle(eds_display, (self.ix, self.iy),
                              (x, y), (0, 255, 0), 1)
                cv2.imshow(
                    self.window_name,
                    eds_display)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.fx, self.fy = x, y
            if abs(self.ix - self.fx) > 5 and abs(self.iy - self.fy) > 5:
                cv2.rectangle(self.eds_copy, (self.ix, self.iy),
                              (self.fx, self.fy), (0, 255, 0), 2)
                cv2.imshow(
                    self.window_name,
                    self.eds_copy)

                # Calculate statistics for the selected region
                self.calculate_statistics(self.ix, self.iy, self.fx, self.fy)

                # Enable threshold controls
                self.bulk_selected = True
                self.enable_threshold_controls()

    def select_exclusion_areas(self):
        if self.sem_data.shape == (0, 0):
            messagebox.showerror("Error", "No SEM image loaded")
            return

        # Reset exclusion data
        #  self.exclusion_rects = []
        self.current_exclusion_rect = np.zeros((0, 0), dtype=np.uint8)

        # Create a copy of SEM image with current masks applied for display
        sem_display = self.sem_data.copy()
        if self.sem_mask.shape != (0, 0):
            # Red overlay for existing mask
            sem_display[self.sem_mask] = [0, 0, 255]

        self.ex_window_name = \
            'Select areas to exclude from analysis (Press Q to finish)'

        cv2.namedWindow(self.ex_window_name)
        cv2.setMouseCallback(self.ex_window_name,
                             self.draw_rectangle_exclusion)
        cv2.imshow(self.ex_window_name, sem_display)

        messagebox.showinfo("Instructions",
                            "Click and drag to select rectangular areas to \
                                    exclude from analysis.\n"
                            "You can select multiple areas. Press 'Q' when \
                                    finished.")

        self.exclusion_working_image = sem_display.copy()

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or not \
                    cv2.getWindowProperty(self.ex_window_name,
                                          cv2.WND_PROP_VISIBLE):
                break

        cv2.destroyAllWindows()

        # Create exclusion mask from all rectangles
        self.create_exclusion_mask()

        # Update the display
        self.apply_masks()

    def draw_rectangle_exclusion(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_exclusion_rect = [x, y, x, y]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_exclusion_rect[2] = x
                self.current_exclusion_rect[3] = y

                # Create display image with all rectangles
                display_img = self.exclusion_working_image.copy()

                # Draw all existing exclusion rectangles in blue
                for rect in self.exclusion_rects:
                    cv2.rectangle(
                        display_img, (rect[0], rect[1]), (rect[2], rect[3]),
                        (255, 0, 0), 2)

                # Draw current rectangle being drawn
                if self.current_exclusion_rect:
                    cv2.rectangle(display_img,
                                  (self.current_exclusion_rect[0],
                                   self.current_exclusion_rect[1]),
                                  (self.current_exclusion_rect[2],
                                   self.current_exclusion_rect[3]),
                                  (255, 0, 0), 1)

                cv2.imshow(self.ex_window_name, display_img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if (abs(self.current_exclusion_rect[0] -
                    self.current_exclusion_rect[2]) > 5 and
                    abs(self.current_exclusion_rect[1] -
                        self.current_exclusion_rect[3]) > 5):

                # Add the rectangle to our list
                self.exclusion_rects.append(self.current_exclusion_rect.copy())

                # Update display image with the new rectangle
                cv2.rectangle(self.exclusion_working_image,
                              (self.current_exclusion_rect[0],
                               self.current_exclusion_rect[1]),
                              (self.current_exclusion_rect[2],
                               self.current_exclusion_rect[3]),
                              (255, 0, 0), 2)

                cv2.imshow(self.ex_window_name, self.exclusion_working_image)

    def create_exclusion_mask(self):
        """Create a mask from all exclusion rectangles"""
        if not self.exclusion_rects and \
                self.sem_data.shape != (0, 0):
            self.exclusion_mask = np.zeros((0, 0), dtype=np.uint8)
            self.exclusion_info_label.config(
                text="No exclusion areas selected", fg="gray")
            self.clear_exclusions_button.config(state=tk.DISABLED)
            return

        # Initialize exclusion mask
        self.exclusion_mask = np.zeros(self.sem_data.shape[:2], dtype=bool)

        # Add each rectangle to the mask
        for rect in self.exclusion_rects:
            x1, y1, x2, y2 = rect
            # Ensure coordinates are ordered correctly
            x_start, x_end = sorted([x1, x2])
            y_start, y_end = sorted([y1, y2])

            # Ensure coordinates are within bounds
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(self.sem_data.shape[1], x_end)
            y_end = min(self.sem_data.shape[0], y_end)

            self.exclusion_mask[y_start:y_end, x_start:x_end] = True

        # Update info label
        total_pixels = self.exclusion_mask.sum()
        total_area = self.sem_data.shape[0] * self.sem_data.shape[1]
        percentage = (total_pixels / total_area) * 100

        self.exclusion_info_label.config(
            text=f"{len(self.exclusion_rects)} exclusion area(s) selected "
                 f"({total_pixels} pixels, {percentage:.1f}%)", fg="black")

        self.clear_exclusions_button.config(state=tk.NORMAL)

    def clear_exclusions(self):
        """Clear all exclusion areas"""
        self.exclusion_rects = []
        self.exclusion_mask = np.zeros((0, 0), dtype=np.uint8)
        self.exclusion_info_label.config(
            text="No exclusion areas selected", fg="gray")
        self.clear_exclusions_button.config(state=tk.DISABLED)

        # Update masks and display
        if self.bulk_selected:
            self.update_masks()

    def calculate_statistics(self, x1, y1, x2, y2):
        # Ensure coordinates are ordered correctly
        x_start, x_end = sorted([x1, x2])
        y_start, y_end = sorted([y1, y2])

        # Extract the region of interest from original EDS data
        roi = self.eds_original[y_start:y_end, x_start:x_end]

        if CYTHON_AVAILABLE:
            # Use Cython optimized statistics calculation
            self.mean_eds, self.std_eds = calculate_region_statistics_fast(
                roi, self.max_eds
            )
        else:
            # Fall back to original NumPy implementation
            self.mean_eds = np.mean(roi) / 255 * self.max_eds
            self.std_eds = np.std(roi) / 255 * self.max_eds

        # Update GUI labels
        self.mean_label.config(text=f"{self.mean_eds:.2f}", fg="black")
        self.std_label.config(text=f"{self.std_eds:.2f}", fg="black")

        print(f"\nSelected Region Statistics:")
        print(f"Region size: {roi.shape[1]}x{roi.shape[0]} pixels")
        print(f"Average pixel value: {self.mean_eds:.2f}")
        print(f"Standard deviation: {self.std_eds:.2f}")


def main():
    root = tk.Tk()
    app = EDSAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
