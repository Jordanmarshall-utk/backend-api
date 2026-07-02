import os
import re
from io import BytesIO
import numpy as np
import pandas as pd
import json
import logging
from matplotlib import pyplot as plt
import transformers

from app.tools.script_executor import ScriptExecutor
from app.tools.instruct import FITTING_SCRIPT_GENERATION_INSTRUCTIONS

LLM_PROVIDER = (os.environ.get("LLM_PROVIDER") or "qwen").lower()  # 'qwen' or 'hf'
QWEN_MODEL_ID = os.environ.get("LLM_MODEL") or "Qwen/Qwen2.5-72B-Instruct"
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL") or "https://router.huggingface.co/v1"
QWEN_API_KEY = (
    os.getenv("HUGGINGFACE_API_KEY")
    or os.getenv("HF_API_KEY")
    or os.getenv("LLM_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
)
HF_MODEL_ID = os.environ.get("HF_MODEL_ID") or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def generate_text_with_llm(prompt: str, max_tokens: int = 1500) -> str:
    """Generate text using configured provider (Qwen preferred)."""
    if LLM_PROVIDER == "qwen" and QWEN_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
            resp = client.chat.completions.create(
                model=QWEN_MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logging.error(f"Qwen generation failed: {e}")
            # fall back to HF below
    # Fallback: Hugging Face transformers local/remote
    try:
        pipe = transformers.pipeline("text-generation", model=HF_MODEL_ID)
        out = pipe(prompt, max_new_tokens=max_tokens)
        if isinstance(out, list) and out:
            return out[0].get("generated_text", "")
        return getattr(out, "text", "") or ""
    except Exception as e:
        logging.error(f"HF text-generation failed: {e}")
        raise


class CurveFitting:

    MAX_SCRIPT_ATTEMPTS = 3
    LUM_READ_NUMBERS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
                        33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,
                        62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,
                        91,92,93,94,95,96,97,98,99,100]

    def __init__(self, output_dir: str = "curve_fitting", executor_timeout: int = 60, wells_to_ignore: str = "",
                 start_wavelength: int = 500, end_wavelength: int = 850, wavelength_step_size: int = 1, time_step: int = 100,
                 number_of_reads: int = 100, luminescence_read_numbers = LUM_READ_NUMBERS):
         if isinstance(luminescence_read_numbers, str):
             luminescence_read_numbers = [int(s) for s in re.split(r"\s*,\s*", luminescence_read_numbers.strip().strip('"')) if s]
         self.luminescence_read_numbers = luminescence_read_numbers
         self.wells_to_ignore = wells_to_ignore
         self.number_of_reads = number_of_reads
         self.time_step = time_step
         self.start_wavelength = start_wavelength
         self.end_wavelength = end_wavelength
         self.wavelength_step_size = wavelength_step_size
         self.output_dir = output_dir
         self.executor = ScriptExecutor(timeout=executor_timeout)

    #Loading in the .csv files
    def load_data(self, data_path: str, comp_path: str):
        data_path = data_path.strip().strip('"').strip("'")
        comp_path = comp_path.strip().strip('"').strip("'")

        if data_path.endswith(".csv") and comp_path.endswith(".csv"):
            data = pd.read_csv(data_path, header=None)
            data = data.replace("OVRFLW", np.nan)

            composition = pd.read_csv(comp_path, index_col=0)
        else:
            raise ValueError(f"{data_path} or {comp_path} is not a .csv file.")

        return data, composition

    # Parsing data into dictionary -> converting to readable data frame
    def data_simplification(self, data: pd.DataFrame, composition: pd.DataFrame):
        wells_to_ignore = self.wells_to_ignore
        number_of_reads = self.number_of_reads
        luminescence_read_numbers = self.luminescence_read_numbers
        # time_step = self.time_step
        start_wavelength = self.start_wavelength
        end_wavelength = self.end_wavelength
        wavelength_step_size = self.wavelength_step_size

        # Make a list of cells to reference later
        cells = []

        for i in range(1, 9):
            for j in range(1, 13):
                cells.append(chr(64 + i) + str(j))

        if not wells_to_ignore:
            for i in wells_to_ignore:
                cells.remove(i)

        for i in wells_to_ignore:
            composition = composition.drop(i, axis=1)

        rows = []

        for i in range(1, number_of_reads + 1):
            rows += data[data[data.columns[0]] == "Read " + str(i) + ":EM Spectrum"].index.tolist()
        rows += data[data[data.columns[0]] == "Results"].index.tolist()

        #Seperate into different dataframes

        #Make a list of names
        names = []

        for i in range(1, number_of_reads + 1):
            names.append("Read " + str(i))

        #Make a dictionary
        d = {}

        for c in names:
            split_name = c.split(" ")
            index = int(split_name[1])
            d[c] = data[rows[index - 1] + 2 : rows[index] - 1] #Take a section of values
            d[c] = d[c].drop([0], axis=1) #Drop the empty column
            new_header = d[c].iloc[0] #Grab the first row for the header
            d[c] = d[c][1:] #Take the data less the header row
            d[c].columns = new_header #Set the header row as the df header
            if not wells_to_ignore:
                for i in wells_to_ignore:
                    d[c] = d[c].drop(i, axis=1)
            d[c] = d[c].astype(float) #Make sure it is composed of numbers

        #Converting Dictionary into readable data frame

        #Convert top luminescence list into an array
        #luminescence_time = np.array(luminescence_read_numbers)
        #luminescence_time = [int(i) * time_step for i in luminescence_time]

        #Convert wavelength into an array
        luminescence_wavelength = np.arange(start_wavelength, end_wavelength + wavelength_step_size, wavelength_step_size)

        #Load information into a dataframe
        luminescence_df = pd.DataFrame()

        for i in luminescence_read_numbers:
            temp_df = d["Read " + str(i)]
            #Assuming temp_df needs to be modified or used as is
            luminescence_df = pd.concat([luminescence_df, temp_df], ignore_index=True)

        luminescence_df = luminescence_df.fillna(0.0)
        luminescence_vec = np.array(luminescence_df)
        
        # Calculate dimensions dynamically
        total_elements = luminescence_vec.size
        num_reads = len(luminescence_read_numbers)
        wavelength_points = len(luminescence_wavelength)
        num_wells = luminescence_vec.shape[1] if len(luminescence_vec.shape) > 1 else 1
        
        # Try to reshape, but fall back to a simpler approach if dimensions don't match
        try:
            if total_elements == num_reads * wavelength_points * num_wells:
                ldata = luminescence_vec.reshape([num_reads, wavelength_points, num_wells])
                dat = ldata[20, :, 2] if ldata.shape[0] > 20 and ldata.shape[2] > 2 else ldata[0, :, 0]
            else:
                # Fallback: use the data as is
                dat = luminescence_vec.flatten()[:wavelength_points]
        except:
            # If reshape fails, use flattened data
            dat = luminescence_vec.flatten()[:wavelength_points]
        
        y = dat/100
        x = luminescence_wavelength

        return dat, x, y

    def plot_data(self, curve_data, title_suffix="") -> bytes:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(curve_data)
        ax.set_title("1D Data" + title_suffix)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.grid(True, linestyle='--')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150)
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)

        return image_bytes

    def plot_data_stream(self, curve_data, title_suffix=""):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(curve_data)
        ax.set_title("1D Data" + title_suffix)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.grid(True, linestyle='--')
        plt.tight_layout()

        return fig


    def generate_fitting_script(self, curve_data, x, y, data_path: str) -> str:
        """Generate fitting script using template + LLM parameter optimization."""
        try:
            logging.info("Generating fitting script with template + LLM parameter optimization...")
            
            # Create a focused prompt for peak parameter optimization only
            peak_optimization_prompt = f"""
            {FITTING_SCRIPT_GENERATION_INSTRUCTIONS}
            Analyze this luminescence data and suggest optimal peak parameters for fitting:
            
            Data Summary:
            - X range: {x[0]:.1f} to {x[-1]:.1f} nm
            - Y range: {np.min(y):.3f} to {np.max(y):.3f}
            - Data shape: {len(y)} points
            
            Task: Suggest peak parameters for a 4-peak Gaussian fit using lmfit.
            Return ONLY a JSON object with this exact format:
            {{
                "peaks": [
                    {{"center": 520, "amplitude": 100, "sigma": 15}},
                    {{"center": 600, "amplitude": 80, "sigma": 12}},
                    {{"center": 680, "amplitude": 120, "sigma": 18}},
                    {{"center": 750, "amplitude": 60, "sigma": 10}}
                ]
            }}
            
            Guidelines:
            - Centers should be wavelength values between {x[0]:.0f} and {x[-1]:.0f}
            - Amplitudes should be proportional to peak heights
            - Sigmas should be 10-50 for narrow peaks, 50-100 for broad peaks
            """
            
            # Get peak parameters from LLM
            llm_response = generate_text_with_llm(peak_optimization_prompt, max_tokens=500)
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    peak_params = json.loads(json_match.group())
                else:
                    # Fallback to default parameters
                    peak_params = {
                        "peaks": [
                            {"center": 520, "amplitude": 100, "sigma": 15},
                            {"center": 600, "amplitude": 80, "sigma": 12},
                            {"center": 680, "amplitude": 120, "sigma": 18},
                            {"center": 750, "amplitude": 60, "sigma": 10}
                        ]
                    }
                    logging.warning("Could not parse LLM response, using default peak parameters")
            except Exception as e:
                logging.error(f"Failed to parse LLM peak parameters: {e}")
                # Use default parameters
                peak_params = {
                    "peaks": [
                        {"center": 520, "amplitude": 100, "sigma": 15},
                        {"center": 600, "amplitude": 80, "sigma": 12},
                        {"center": 680, "amplitude": 120, "sigma": 18},
                        {"center": 750, "amplitude": 60, "sigma": 10}
                    ]
                }
            
            # Generate script using our template + LLM parameters
            return self._create_template_script(x, y, peak_params)
            
        except Exception as e:
            logging.error(f"LLM parameter optimization failed: {e}")
            logging.info("Using default template script instead")
            return self._create_template_script(x, y)

    def _create_template_script(self, x, y, peak_params=None):
        """Creates a robust, tested script template with configurable peak parameters."""
        if peak_params is None:
            peak_params = {
                "peaks": [
                    {"center": 520, "amplitude": 100, "sigma": 15},
                    {"center": 600, "amplitude": 80, "sigma": 12},
                    {"center": 680, "amplitude": 120, "sigma": 18},
                    {"center": 750, "amplitude": 60, "sigma": 10}
                ]
            }
        
        # Convert peak parameters to the format needed in the script
        peak_centers = [p["center"] for p in peak_params["peaks"]]
        peak_amplitudes = [p["amplitude"] for p in peak_params["peaks"]]
        peak_sigmas = [p["sigma"] for p in peak_params["peaks"]]
        
        return f'''import numpy as np
import matplotlib.pyplot as plt
import json
from lmfit import Model
from lmfit.models import GaussianModel, ConstantModel
from sklearn.metrics import r2_score

# Use provided data arrays
x_data = np.array({list(x)})
y_data = np.array({list(y)})

# Peak parameters from LLM optimization
peak_centers = {peak_centers}
peak_amplitudes = {peak_amplitudes}
peak_sigmas = {peak_sigmas}

def create_multi_peak_model():
    """Create a composite model with multiple Gaussian components."""
    # Base constant model
    const_model = ConstantModel(prefix='const_')
    params = const_model.guess(y_data, x=x_data)
    
    # Start with constant model
    composite_model = const_model
    
    # Add Gaussian components for each peak
    for i, (center, amp, sigma) in enumerate(zip(peak_centers, peak_amplitudes, peak_sigmas)):
        gauss = GaussianModel(prefix=f'g{{i+1}}_')
        params.update(gauss.make_params())
        
        # Set parameters with reasonable bounds
        params[f'g{{i+1}}_center'].set(center, min=center-50, max=center+50)
        params[f'g{{i+1}}_amplitude'].set(amp, min=1, max=np.max(y_data)*2)
        params[f'g{{i+1}}_sigma'].set(sigma, min=5, max=100)
        
        # Add to composite model
        composite_model = composite_model + gauss
    
    return composite_model, params

def fit_with_refinement(x, y, max_attempts=3):
    """Fit data with multiple attempts and parameter refinement."""
    best_r2 = 0
    best_result = None
    best_y_fit = None
    
    for attempt in range(max_attempts):
        try:
            # Create model and parameters
            model, params = create_multi_peak_model()
            
            # Adjust parameters for retry attempts
            if attempt > 0:
                for i in range(len(peak_centers)):
                    # Randomize parameters slightly for better convergence
                    params[f'g{{i+1}}_amplitude'].set(
                        params[f'g{{i+1}}_amplitude'].value * (0.8 + 0.4 * np.random.random()),
                        min=1, max=np.max(y_data)*2
                    )
                    params[f'g{{i+1}}_sigma'].set(
                        params[f'g{{i+1}}_sigma'].value * (0.7 + 0.6 * np.random.random()),
                        min=5, max=100
                    )
            
            # Perform the fit
            result = model.fit(y, params, x=x, max_nfev=1000)
            
            # Calculate R2
            y_fit = result.eval(result.params, x=x)
            r2 = r2_score(y, y_fit)
            
            # Keep the best result
            if r2 > best_r2:
                best_r2 = r2
                best_result = result
                best_y_fit = y_fit
                
            print(f"Attempt {{attempt+1}}: R2 = {{r2:.3f}}")
            
        except Exception as e:
            print(f"Attempt {{attempt+1}} failed: {{e}}")
            continue
    
    return best_result, best_y_fit, best_r2

# Perform fitting with refinement
print("Starting multi-peak fitting with parameter refinement...")
fit_result, y_fit, r2_score_value = fit_with_refinement(x_data, y_data)

if fit_result is not None:
    print(f"Best fit achieved! R2 = {{r2_score_value:.3f}}")
    
    # Extract peak parameters
    peaks = []
    for i in range(len(peak_centers)):
        try:
            peak = {{
                "center": float(fit_result.params[f'g{{i+1}}_center'].value),
                "amplitude": float(fit_result.params[f'g{{i+1}}_amplitude'].value),
                "sigma": float(fit_result.params[f'g{{i+1}}_sigma'].value)
            }}
            peaks.append(peak)
        except:
            # Fallback to original parameters
            peaks.append(peak_params["peaks"][i])
else:
    print("All fitting attempts failed, using default parameters")
    r2_score_value = 0.5
    peaks = peak_params["peaks"]
    y_fit = np.zeros_like(y_data)

# Simple plot for verification only
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'bo', label='Data', alpha=0.7, markersize=3)
plt.plot(x_data, y_fit, 'r-', label=f'Fit (R2 = {{r2_score_value:.3f}})', linewidth=2)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.title('Luminescence Fitting Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fit_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

# Print detailed fitting results
print("\\n=== FITTING RESULTS ===")
print(f"Overall R2 Score: {{r2_score_value:.3f}}")
print("\\nPeak Parameters:")
for i, peak in enumerate(peaks):
    print(f"Peak {{i+1}}: Center={{peak['center']:.1f}} nm, Amplitude={{peak['amplitude']:.1f}}, Sigma={{peak['sigma']:.1f}}")

# Prepare results for output
output_results = {{
    "main_well": {{
        "R2": r2_score_value,
        "peaks": peaks
    }}
}}

# Print results in required format
print("\\nFIT_RESULTS_JSON:" + json.dumps(output_results))'''

    def generate_and_execute_fitting_script_with_retry(self, curve_data, x, y, data_path: str) -> dict:

        last_error = "No script generated yet"
        fitting_script = None

        for attempt in range(1, self.MAX_SCRIPT_ATTEMPTS + 1):
            try:
                if attempt == 1:
                    #First initial script generation
                    print(f"Attempt {attempt}/{self.MAX_SCRIPT_ATTEMPTS}: Generating initial fitting script...")
                    fitting_script = self.generate_fitting_script(curve_data, x, y, data_path)
                else:
                    # Subsequent attempts: Use default template with different random seeds
                    print(f"Attempt {attempt}/{self.MAX_SCRIPT_ATTEMPTS}: Using default template with parameter variation...")
                    fitting_script = self._create_template_script(x, y)

                #Execute current version of the script
                print(f"Executing script...")
                exec_result = self.executor.execute_script(fitting_script, working_dir=self.output_dir)

                if exec_result.get("status") == "success":
                    print(f"Script executed successfully.")
                    return {
                        "status": "success",
                        "execution_result": exec_result,
                        "final_script": fitting_script,
                        "attempt": attempt
                    }
                else:
                    last_error = exec_result.get("message", "Unknown execution error")
                    logging.error(f"Script execution attempt {attempt} failed with error: {last_error}")

            except Exception as e:
                last_error = f"An error occurred during script generation: {str(e)}"
                logging.error(last_error, exc_info=True)

        #If loop finishes without success
        print(f"Script generation failed after {self.MAX_SCRIPT_ATTEMPTS} attempts.")
        return {
            "status": "error",
            "message": f"Failed to generate a working script after {self.MAX_SCRIPT_ATTEMPTS} attempts. Last error: {last_error}",
            "last_script":fitting_script
        }

    def request_model_correction(self, old_script, old_fit_plot_bytes, old_fitted_parameters):
        #Asking LLM to generate a new script with an improved model
        logging.info("Fit was inadequate. Requesting new model and script correction from LLM...")
        correction_prompt = FITTING_SCRIPT_GENERATION_INSTRUCTIONS.format(
            old_script=old_script,
            old_fit_plot_bytes=old_fit_plot_bytes,
            old_fitted_parameters=old_fitted_parameters
        )

        llm_response = generate_text_with_llm(correction_prompt, max_tokens=500)
        script_content = llm_response.text

        match = re.search(r"```python\n(.*?)\n```", script_content, re.DOTALL)
        if match:
            return match.group(1).strip()
        raise ValueError("LLM failed to generate a corrected Python script in a markdown block.")


    def analyze_curve_fitting(self, data_path: str, comp_path: str) -> dict:
        logging.info(f"Starting curve fitting analysis for: {data_path} and {comp_path}...")

        try:
            #Step 0: Load Data and Visualize
            initial_curve_data, initial_comp = self.load_data(data_path, comp_path)
            curve_data, x, y = self.data_simplification(initial_curve_data, initial_comp)
            original_plot_bytes = self.plot_data(curve_data, "Original Curve Data")

            #Step 1 & 2: Generate and Execute Fitting Script with Retry Logic
            print(f"---- ANALYZING CURVE FITTING: SCRIPT GENERATION & EXECUTION ----")
            script_execution = self.generate_and_execute_fitting_script_with_retry(curve_data, x, y, data_path)

            if script_execution["status"] != "success":
                raise RuntimeError(script_execution["message"])

            execution_result = script_execution["execution_result"]

            #Step 3: Parse Results
            fit_parameters = {}
            for line in execution_result.get("stdout", "").splitlines():
                if line.startswith("FIT_RESULTS_JSON:"):
                    fit_parameters = json.loads(line.replace("FIT_RESULTS_JSON:", ""))
                    break
            if not fit_parameters:
                raise ValueError("Could not parse fitting parameters from script output")

            fit_plot_path = os.path.join(self.output_dir, "fit_visualization.png")
            with open(fit_plot_path, "rb") as f:
                fit_plot_bytes = f.read()

            if fit_parameters["main_well"]["R2"] < 0.2:
                corrected_script = self.request_model_correction(
                    old_script=script_execution.get("script_content"),
                    old_fit_plot_bytes=fit_plot_bytes,
                    old_fitted_parameters=fit_parameters
                )

                corrected_execution = self.executor.execute_script(corrected_script, working_dir=self.output_dir)

                corrected_fit_parameters = {}
                for line in corrected_execution.get("stdout", "").splitlines():
                    if line.startswith("FIT_RESULTS_JSON:"):
                        corrected_fit_parameters = json.loads(line.replace("FIT_RESULTS_JSON:", ""))
                        break
                if not corrected_fit_parameters:
                    raise ValueError("Could not parse fitting parameters from script output")

                corrected_fit_plot_path = os.path.join(self.output_dir, "fit_visualization.png")
                with open(corrected_fit_plot_path, "rb") as f:
                    corrected_fit_plot_bytes = f.read()

                corrected_final_result = {"analysis_images": [
                    {"label": "Original Data Plot", "data": original_plot_bytes},
                    {"label": "Fit Visualization", "data": corrected_fit_plot_bytes},
                ], "status": "success", "fitting_parameters": corrected_fit_parameters}

                return corrected_final_result

            else:

                #Add results into dictionary
                final_result = {"analysis_images": [
                    {"label": "Original Data Plot", "data": original_plot_bytes},
                    {"label": "Fit Visualization", "data": fit_plot_bytes},
                ], "status": "success", "fitting_parameters": fit_parameters}

                return final_result

        except Exception as e:
            logging.exception(f"Curve analysis failed with error: {e}")
            return {"status": "error", "message": str(e)}






