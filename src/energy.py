import requests
import time

# URL of the LibreHardwareMonitor web server
URL = "http://localhost:8085/data.json"

def parse_value(val_str):
    """
    Cleans strings like '45.5 °C' or '12.1 W' into a float.
    Handles '45,5' for European locales.
    """
    try:
        # Remove units and normalize decimal points
        clean = val_str.replace(" W", "").replace(" °C", "").replace(" %", "").replace(",", ".")
        return float(clean)
    except:
        return 0.0

def get_stats():
    try:
        response = requests.get(URL, timeout=1)
        data = response.json()
        
        stats = {
            "cpu_pwr": 0.0, "cpu_temp": 0.0,
            "gpu_pwr": 0.0, "gpu_temp": 0.0
        }
        
        # Traverse: Computer -> Hardware -> Sensors
        if 'Children' in data:
            for computer_node in data['Children']:
                for hardware in computer_node['Children']:
                    
                    # --- 1. CPU DETECTION ---
                    # Look for Intel or AMD in the name, or a generic 'Cpu' image icon
                    if "Intel" in hardware['Text'] or "AMD" in hardware['Text'] or "Cpu" in hardware['ImageURL']:
                        for group in hardware['Children']:
                            
                            # CPU TEMP
                            if "Temperatures" in group['Text']:
                                for sensor in group['Children']:
                                    # Preference: 'Package' temp, fallback to 'Core Max'
                                    if "Package" in sensor['Text'] or "Core Max" in sensor['Text']:
                                        stats['cpu_temp'] = parse_value(sensor['Value'])

                            # CPU POWER
                            if "Powers" in group['Text']:
                                for sensor in group['Children']:
                                    if "Package" in sensor['Text']:
                                        stats['cpu_pwr'] = parse_value(sensor['Value'])

                    # --- 2. GPU DETECTION ---
                    # Look for NVIDIA, Radeon, or 'Gpu' image icon
                    if "NVIDIA" in hardware['Text'] or "Radeon" in hardware['Text'] or "Gpu" in hardware['ImageURL']:
                        for group in hardware['Children']:
                            
                            # GPU TEMP
                            if "Temperatures" in group['Text']:
                                for sensor in group['Children']:
                                    if "Core" in sensor['Text']:
                                        stats['gpu_temp'] = parse_value(sensor['Value'])

                            # GPU POWER
                            if "Powers" in group['Text']:
                                for sensor in group['Children']:
                                    # 'Package', 'Board', or just 'GPU Power'
                                    if "Package" in sensor['Text'] or "Power" in sensor['Text']:
                                        stats['gpu_pwr'] = parse_value(sensor['Value'])
        return stats

    except Exception as e:
        print(f"Connection Error: {e}")
        return None

if __name__ == "__main__":
    print(f"Connecting to {URL}...")
    print(f"{'CPU PWR':<10} | {'CPU TEMP':<10} || {'GPU PWR':<10} | {'GPU TEMP':<10}")
    print("-" * 50)

    while True:
        data = get_stats()
        
        if data:
            print(f"{data['cpu_pwr']:6.1f} W   | {data['cpu_temp']:6.1f} °C   || {data['gpu_pwr']:6.1f} W   | {data['gpu_temp']:6.1f} °C")
        else:
            print("Is LibreHardwareMonitor Web Server running?")
            
        time.sleep(1)