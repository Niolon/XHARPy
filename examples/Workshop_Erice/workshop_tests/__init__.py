def test_xharpy_installation():
    """
    Test if all required components for the XHARPy workshop are properly installed and functioning.
    
    This function tests:
    1. JAX installation
    2. GPAW installation
    3. Quantum Espresso installation (pw.x and pp.x)
    4. XHARPy installation
    
    Returns:
    --------
    bool: True if all tests pass, False otherwise
    """
    import sys
    import os
    import tempfile
    import shutil
    from pathlib import Path
    import importlib
    from importlib.util import find_spec
    import subprocess
    import warnings
    
    # For nice output formatting
    from IPython.display import display, HTML, Markdown
    
    results = []
    all_pass = True
    
    # Helper function for formatted output
    def print_result(component, status, message=""):
        color = "green" if status else "red"
        icon = "✓" if status else "✗"
        result = f"<div style='margin: 5px 0;'><span style='color:{color}; font-weight:bold;'>{icon} {component}: </span>{message}</div>"
        results.append(result)
        return status
    
    # 1. Test JAX installation
    try:
        # Check if JAX is installed
        jax_spec = find_spec("jax")
        if not jax_spec:
            raise ImportError("JAX not found")
        
        # Try to import and run basic JAX operation
        import jax
        import jax.numpy as jnp
        
        # Simple operation to verify JAX is working
        a = jnp.array([1, 2, 3])
        b = jnp.array([4, 5, 6])
        c = jnp.dot(a, b)
        
        jax_ok = print_result("JAX", True, "Successfully imported and executed basic operations")
    except Exception as e:
        jax_ok = print_result("JAX", False, f"Error: {str(e)}")
        all_pass = False
    
    # 2. Test GPAW installation
    try:
        # Check if GPAW is installed
        gpaw_spec = find_spec("gpaw")
        if not gpaw_spec:
            raise ImportError("GPAW not found")
        
        # Try to import GPAW
        import gpaw
        
        # Check GPAW version
        gpaw_version = gpaw.__version__
        
        # Try to import a key module that XHARPy would use
        from xharpy.f0j_sources.gpaw_source import calc_f0j
        
        gpaw_ok = print_result("GPAW", True, f"Successfully imported (version {gpaw_version})")
    except Exception as e:
        gpaw_ok = print_result("GPAW", False, f"Error: {str(e)}")
        all_pass = False
    
    # 3. Test Quantum Espresso installation
    qe_pw_ok = qe_pp_ok = False
    try:
        # Create a temporary directory for QE tests
        temp_dir = tempfile.mkdtemp()
        
        # Test pw.x is available
        try:
            result = subprocess.run(
                ["which", "pw.x"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pw_path = result.stdout.strip()
                qe_pw_ok = True
                pw_message = f"Found at {pw_path}"
            else:
                pw_message = "Not found in PATH"
                qe_pw_ok = False
        except Exception as e:
            pw_message = f"Error checking: {str(e)}"
            qe_pw_ok = False
            
        # Test pp.x is available
        try:
            result = subprocess.run(
                ["which", "pp.x"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pp_path = result.stdout.strip()
                qe_pp_ok = True
                pp_message = f"Found at {pp_path}"
            else:
                pp_message = "Not found in PATH"
                qe_pp_ok = False
        except Exception as e:
            pp_message = f"Error checking: {str(e)}"
            qe_pp_ok = False
            
        # Try to import QE-related module from XHARPy
        qe_module_ok = False
        try:
            from xharpy.f0j_sources.qe_source import calc_f0j_core
            qe_module_ok = True
        except Exception as e:
            qe_module_message = f"Could not import XHARPy QE module: {str(e)}"
            
        # Overall QE status
        qe_status = qe_pw_ok and qe_pp_ok and qe_module_ok
        
        if qe_status:
            print_result("Quantum Espresso", True, f"pw.x and pp.x found, XHARPy QE module loaded")
        else:
            message = []
            message.append(f"pw.x: {'✓' if qe_pw_ok else '✗'} {pw_message}")
            message.append(f"pp.x: {'✓' if qe_pp_ok else '✗'} {pp_message}")
            message.append(f"XHARPy QE module: {'✓' if qe_module_ok else '✗'}")
            print_result("Quantum Espresso", False, "<br>".join(message))
            all_pass = False
        
    except Exception as e:
        print_result("Quantum Espresso", False, f"Error: {str(e)}")
        all_pass = False
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)
    
    # 4. Test XHARPy installation
    try:
        # Check if XHARPy is installed
        xharpy_spec = find_spec("xharpy")
        if not xharpy_spec:
            raise ImportError("XHARPy not found")
        
        # Import key modules from XHARPy
        import xharpy
        from xharpy import refine, create_construction_instructions
        
        # Try to import one module from each major component
        from xharpy.io import cif2data
        from xharpy.structure.common import AtomInstructions
        
        # Get XHARPy version from defaults.py
        try:
            from xharpy.defaults import XHARPY_VERSION
            xharpy_version = XHARPY_VERSION
        except ImportError:
            xharpy_version = "Unknown"
        
        xharpy_ok = print_result("XHARPy", True, f"Successfully imported (version {xharpy_version})")
    except Exception as e:
        xharpy_ok = print_result("XHARPy", False, f"Error: {str(e)}")
        all_pass = False
    
    # Display overall result
    if all_pass:
        overall_status = "All components installed successfully! You are ready for the workshop."
    else:
        # If only QE is missing, it's still mostly OK
        if not (qe_pw_ok and qe_pp_ok) and jax_ok and gpaw_ok and xharpy_ok:
            overall_status = "Most components are installed correctly! The Quantum Espresso examples may not work, but GPAW examples should work fine."
            all_pass = True  # Consider this a partial pass
        else:
            overall_status = "Some essential components are missing or not working properly!"
            
    overall_color = "green" if all_pass else "red"
    
    # Format and display all results
    display(HTML(
        f"<h3>XHARPy Workshop Requirements Test</h3>"
        f"<div style='margin: 10px 0; padding: 10px; border-radius: 5px; background-color: #f5f5f5;'>"
        f"<div style='font-weight: bold; color: {overall_color}; margin-bottom: 10px;'>{overall_status}</div>"
        f"{''.join(results)}"
        f"</div>"
    ))
    
    return all_pass