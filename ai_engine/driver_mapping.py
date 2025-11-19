"""
Driver Name Mapping Utility
Maps vehicle IDs to actual driver names
"""

import pandas as pd
import os

# Static mapping from vehicle_number to full vehicle_id
VEHICLE_ID_MAP = {
    '0': 'GR86-002-000', '2': 'GR86-060-2', '3': 'GR86-040-3', '5': 'GR86-065-5',
    '7': 'GR86-006-7', '8': 'GR86-012-8', '11': 'GR86-035-11', '13': 'GR86-022-13',
    '16': 'GR86-010-16', '18': 'GR86-030-18', '21': 'GR86-047-21', '31': 'GR86-015-31',
    '46': 'GR86-033-46', '47': 'GR86-025-47', '55': 'GR86-016-55', '57': 'GR86-057-57',
    '72': 'GR86-026-72', '80': 'GR86-013-80', '86': 'GR86-021-86', '88': 'GR86-049-88',
    '89': 'GR86-028-89', '93': 'GR86-038-93', '98': 'GR86-036-98', '113': 'GR86-063-113',
}


class DriverNameMapper:
    """Maps vehicle IDs to driver names"""
    
    def __init__(self):
        self.name_map = {}
        self._load_driver_names()
    
    def _load_driver_names(self):
        """Load driver names from sector analysis file"""
        try:
            # Load from sector analysis which has driver names
            df = pd.read_csv(
                'indianapolis/indianapolis/23_AnalysisEnduranceWithSections_Race 1.CSV',
                sep=';'
            )
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Create mapping from car number to name
            # NOTE: 'NUMBER' column contains the actual car number (e.g., 21, 47, 113)
            # 'DRIVER_NUMBER' is always 1 (not useful)
            if 'NUMBER' in df.columns and 'DRIVER_NAME' in df.columns:
                # Get unique driver-number pairs
                unique_drivers = df[['NUMBER', 'DRIVER_NAME']].drop_duplicates()
                
                for _, row in unique_drivers.iterrows():
                    car_num = str(int(row['NUMBER'])).strip()  # Convert to int to remove leading zeros
                    driver_name = str(row['DRIVER_NAME']).strip()
                    
                    # Map car number to name
                    self.name_map[car_num] = driver_name
                    
                    # Also map full vehicle ID if we have it
                    if car_num in VEHICLE_ID_MAP:
                        vehicle_id = VEHICLE_ID_MAP[car_num]
                        self.name_map[vehicle_id] = driver_name
            
            print(f"[Driver Mapper] Loaded {len(set(self.name_map.values()))} driver names")
            
        except Exception as e:
            print(f"[Driver Mapper] Warning: Could not load driver names: {e}")
            # Fallback to empty map
            self.name_map = {}
    
    def get_driver_name(self, vehicle_id: str) -> str:
        """
        Get driver name from vehicle ID
        
        Args:
            vehicle_id: Either vehicle number (e.g., '113') or full ID (e.g., 'GR86-063-113')
        
        Returns:
            Driver name if found, otherwise returns the vehicle_id
        """
        # Try direct lookup
        if vehicle_id in self.name_map:
            return self.name_map[vehicle_id]
        
        # Try extracting number from full ID (e.g., 'GR86-063-113' -> '113')
        if '-' in str(vehicle_id):
            parts = str(vehicle_id).split('-')
            if len(parts) >= 2:
                # Try last part (e.g., '113')
                last_part = parts[-1]
                if last_part in self.name_map:
                    return self.name_map[last_part]
                
                # Try middle part (e.g., '063' -> '63')
                if len(parts) >= 3:
                    middle_part = parts[1]
                    # Remove leading zeros
                    try:
                        middle_num = str(int(middle_part))
                        if middle_num in self.name_map:
                            return self.name_map[middle_num]
                    except:
                        pass
                
                # Try first number part (e.g., '022' -> '22')
                try:
                    first_num = str(int(parts[1]))
                    if first_num in self.name_map:
                        return self.name_map[first_num]
                except:
                    pass
        
        # Try just extracting any numbers
        import re
        numbers = re.findall(r'\d+', str(vehicle_id))
        for num in numbers:
            # Try with leading zeros removed
            clean_num = str(int(num))
            if clean_num in self.name_map:
                return self.name_map[clean_num]
        
        # Return original if not found
        return vehicle_id
    
    def get_display_name(self, vehicle_id: str, include_id: bool = False) -> str:
        """
        Get formatted display name
        
        Args:
            vehicle_id: Vehicle ID
            include_id: If True, includes vehicle ID in parentheses
        
        Returns:
            Formatted name like "Michael Edwards" or "Michael Edwards (#113)"
        """
        driver_name = self.get_driver_name(vehicle_id)
        
        # If name is same as ID, just return it
        if driver_name == vehicle_id:
            return vehicle_id
        
        # If include_id, add vehicle number in parentheses
        if include_id:
            # Extract number from vehicle_id
            if '-' in str(vehicle_id):
                num = str(vehicle_id).split('-')[-1]
                return f"{driver_name} (#{num})"
            else:
                return f"{driver_name} (#{vehicle_id})"
        
        return driver_name


# Global instance
_mapper = None

def get_driver_mapper():
    """Get global driver mapper instance"""
    global _mapper
    if _mapper is None:
        _mapper = DriverNameMapper()
    return _mapper


def get_driver_name(vehicle_id: str) -> str:
    """Convenience function to get driver name"""
    return get_driver_mapper().get_driver_name(vehicle_id)


def get_display_name(vehicle_id: str, include_id: bool = False) -> str:
    """Convenience function to get display name"""
    return get_driver_mapper().get_display_name(vehicle_id, include_id)
