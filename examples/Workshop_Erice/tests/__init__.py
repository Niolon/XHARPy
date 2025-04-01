def test_xharpy_installation():
    """
    Test if all required components for the XHARPy workshop are properly installed and functioning.
    
    This function tests:
    1. JAX installation
    2. GPAW installation
    3. Quantum Espresso installation
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
    
    # For nice output formatting
    from IPython.display import display, HTML, Markdown
    
    results = []
    all_pass = True
    print('ping')
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
        
        # Try to import GPAW and create a simple calculator
        import gpaw
        from gpaw import GPAW
        
        # Check GPAW version
        gpaw_version = gpaw.__version__
        
        gpaw_ok = print_result("GPAW", True, f"Successfully imported (version {gpaw_version})")
    except Exception as e:
        gpaw_ok = print_result("GPAW", False, f"Error: {str(e)}")
        all_pass = False
    
    # 3. Test Quantum Espresso installation
    try:
        # Check if basic QE commands are available
        import subprocess
        
        # Create a temporary directory for QE tests
        temp_dir = tempfile.mkdtemp()
        try:
            # Test pw.x is available by checking version
            result = subprocess.run(
                ["pw.x", "-v"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode != 0:
                raise RuntimeError("Could not execute pw.x")
            
            qe_version = result.stdout.strip() if result.stdout else "Unknown version"
            
            # Try to import QE-related module from XHARPy
            from xharpy.f0j_sources.qe_source import calc_f0j_core
            
            qe_ok = print_result("Quantum Espresso", True, f"Successfully detected QE command line tools")
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)
    except Exception as e:
        qe_ok = print_result("Quantum Espresso", False, f"Error: {str(e)}")
        all_pass = False
    
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
        
        xharpy_version = getattr(xharpy, "XHARPY_VERSION", "Unknown")
        
        xharpy_ok = print_result("XHARPy", True, f"Successfully imported (version {xharpy_version})")
    except Exception as e:
        xharpy_ok = print_result("XHARPy", False, f"Error: {str(e)}")
        all_pass = False
    
    # Display overall result
    overall_status = "All components installed successfully!" if all_pass else "Some components are missing or not working properly!"
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