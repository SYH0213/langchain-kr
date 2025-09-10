import os
import requests
from typing import Optional, Dict, Any
from langchain.tools import BaseTool
from pydantic import Field
import json


class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = """
    Get current weather information for a specific city using SerpAPI.
    Input should be a city name (e.g., 'Seoul', 'New York', 'Paris').
    Returns temperature, weather conditions, humidity, wind speed, and more.
    """
    
    # Use SerpAPI for weather data
    serpapi_key: str = Field(default_factory=lambda: os.getenv("SERPAPI_API_KEY", ""))
    
    def _get_weather_from_serpapi(self, city: str) -> str:
        """Get weather information using SerpAPI."""
        if not self.serpapi_key:
            return "Error: SERPAPI_API_KEY not configured. Please set SERPAPI_API_KEY in .env file"
        
        try:
            # Use SerpAPI to get weather information
            base_url = "https://serpapi.com/search"
            params = {
                "api_key": self.serpapi_key,
                "engine": "google",
                "q": f"weather {city}",
                "location": city,
                "gl": "us",
                "hl": "en"
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if weather data is available in the response
                if "answer_box" in data and "weather" in data.get("answer_box", {}).get("type", ""):
                    weather_data = data["answer_box"]
                    
                    # Extract weather information
                    temperature = weather_data.get("temperature", "N/A")
                    weather_desc = weather_data.get("weather", "N/A")
                    humidity = weather_data.get("humidity", "N/A")
                    wind = weather_data.get("wind", "N/A")
                    precipitation = weather_data.get("precipitation", "N/A")
                    
                    result = f"""
Current weather in {city}:
- Temperature: {temperature}
- Conditions: {weather_desc}
- Humidity: {humidity}
- Wind: {wind}
- Precipitation: {precipitation}
"""
                    return result
                    
                # Fallback to search results if no weather box
                elif "organic_results" in data and len(data["organic_results"]) > 0:
                    weather_info = []
                    for result in data["organic_results"][:3]:
                        snippet = result.get("snippet", "")
                        if any(word in snippet.lower() for word in ["temperature", "°c", "°f", "weather", "celsius"]):
                            weather_info.append(f"• {snippet}")
                    
                    if weather_info:
                        return f"""
Weather information for {city}:

{chr(10).join(weather_info)}

Note: This information is from search results. For more accurate data, weather widget data may not be available for this location.
"""
                    else:
                        return f"Could not find detailed weather information for '{city}'."
                else:
                    return f"Could not find weather information for '{city}'. Please check the city name and try again."
            else:
                return f"Error fetching weather data: HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out. Please try again."
        except Exception as e:
            return f"Error fetching weather data: {str(e)}"
    
    def _run(self, city: str) -> str:
        """Get weather information for a city."""
        # Normalize city name
        city = city.strip()
        
        # Handle Korean city names
        city_mapping = {
            "서울": "Seoul",
            "부산": "Busan",
            "인천": "Incheon",
            "대구": "Daegu",
            "대전": "Daejeon",
            "광주": "Gwangju",
            "울산": "Ulsan",
            "제주": "Jeju"
        }
        
        # Convert Korean city names to English if needed
        city = city_mapping.get(city, city)
        
        # Use SerpAPI to get weather information
        return self._get_weather_from_serpapi(city)
    
    async def _arun(self, city: str) -> str:
        """Async version of weather tool (not implemented)."""
        return self._run(city)