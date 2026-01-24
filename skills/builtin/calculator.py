"""Calculator skill with safe math evaluation."""

import math
import re
from skills import skill


# Safe math functions available in expressions
SAFE_MATH = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'pow': pow,
    'sqrt': math.sqrt,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'asin': math.asin,
    'acos': math.acos,
    'atan': math.atan,
    'log': math.log,
    'log10': math.log10,
    'log2': math.log2,
    'exp': math.exp,
    'floor': math.floor,
    'ceil': math.ceil,
    'pi': math.pi,
    'e': math.e,
    'degrees': math.degrees,
    'radians': math.radians,
}


@skill(
    name="calc",
    description="Evaluate mathematical expressions safely",
    version="1.0.0",
    author="MAUDE",
    triggers=["calculate", "calc", "math", "compute"],
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')"
            }
        },
        "required": ["expression"]
    }
)
def calc(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Clean the expression
        expr = expression.strip()

        # Replace common notations
        expr = expr.replace('^', '**')  # Power notation
        expr = expr.replace('×', '*')
        expr = expr.replace('÷', '/')

        # Validate - only allow safe characters
        allowed = set('0123456789.+-*/()[], ')
        allowed_names = '|'.join(SAFE_MATH.keys())

        # Remove function names for character check
        check_expr = re.sub(rf'\b({allowed_names})\b', '', expr)

        if not all(c in allowed for c in check_expr):
            bad_chars = [c for c in check_expr if c not in allowed]
            return f"Error: Invalid characters in expression: {set(bad_chars)}"

        # Evaluate safely
        result = eval(expr, {"__builtins__": {}}, SAFE_MATH)

        # Format result
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 10)  # Avoid floating point noise

        return f"{expression} = {result}"

    except ZeroDivisionError:
        return "Error: Division by zero"
    except SyntaxError:
        return f"Error: Invalid expression syntax"
    except NameError as e:
        return f"Error: Unknown function or variable: {e}"
    except Exception as e:
        return f"Error: {e}"


@skill(
    name="convert",
    description="Convert between units (length, weight, temperature, data)",
    version="1.0.0",
    author="MAUDE",
    triggers=["convert", "conversion"],
    parameters={
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "Value to convert"
            },
            "from_unit": {
                "type": "string",
                "description": "Source unit"
            },
            "to_unit": {
                "type": "string",
                "description": "Target unit"
            }
        },
        "required": ["value", "from_unit", "to_unit"]
    }
)
def convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units."""

    # Normalize units
    from_u = from_unit.lower().strip()
    to_u = to_unit.lower().strip()

    # Conversion tables (to base unit, then from base unit)
    length_to_m = {
        'm': 1, 'meter': 1, 'meters': 1,
        'km': 1000, 'kilometer': 1000, 'kilometers': 1000,
        'cm': 0.01, 'centimeter': 0.01, 'centimeters': 0.01,
        'mm': 0.001, 'millimeter': 0.001, 'millimeters': 0.001,
        'mi': 1609.344, 'mile': 1609.344, 'miles': 1609.344,
        'yd': 0.9144, 'yard': 0.9144, 'yards': 0.9144,
        'ft': 0.3048, 'foot': 0.3048, 'feet': 0.3048,
        'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
    }

    weight_to_kg = {
        'kg': 1, 'kilogram': 1, 'kilograms': 1,
        'g': 0.001, 'gram': 0.001, 'grams': 0.001,
        'mg': 0.000001, 'milligram': 0.000001, 'milligrams': 0.000001,
        'lb': 0.453592, 'pound': 0.453592, 'pounds': 0.453592,
        'oz': 0.0283495, 'ounce': 0.0283495, 'ounces': 0.0283495,
    }

    data_to_bytes = {
        'b': 1, 'byte': 1, 'bytes': 1,
        'kb': 1024, 'kilobyte': 1024, 'kilobytes': 1024,
        'mb': 1024**2, 'megabyte': 1024**2, 'megabytes': 1024**2,
        'gb': 1024**3, 'gigabyte': 1024**3, 'gigabytes': 1024**3,
        'tb': 1024**4, 'terabyte': 1024**4, 'terabytes': 1024**4,
    }

    # Try each conversion table
    for table, name in [(length_to_m, 'length'), (weight_to_kg, 'weight'), (data_to_bytes, 'data')]:
        if from_u in table and to_u in table:
            base_value = value * table[from_u]
            result = base_value / table[to_u]
            result = round(result, 6) if isinstance(result, float) else result
            return f"{value} {from_unit} = {result} {to_unit}"

    # Temperature (special case)
    temp_result = _convert_temperature(value, from_u, to_u)
    if temp_result:
        return temp_result

    return f"Error: Cannot convert from '{from_unit}' to '{to_unit}'. Check unit names."


def _convert_temperature(value: float, from_u: str, to_u: str) -> str:
    """Convert temperature units."""
    temps = {'c': 'celsius', 'f': 'fahrenheit', 'k': 'kelvin',
             'celsius': 'celsius', 'fahrenheit': 'fahrenheit', 'kelvin': 'kelvin'}

    from_t = temps.get(from_u)
    to_t = temps.get(to_u)

    if not from_t or not to_t:
        return None

    # Convert to Celsius first
    if from_t == 'celsius':
        c = value
    elif from_t == 'fahrenheit':
        c = (value - 32) * 5/9
    elif from_t == 'kelvin':
        c = value - 273.15

    # Convert from Celsius to target
    if to_t == 'celsius':
        result = c
    elif to_t == 'fahrenheit':
        result = c * 9/5 + 32
    elif to_t == 'kelvin':
        result = c + 273.15

    result = round(result, 2)
    return f"{value}°{from_u.upper()[0]} = {result}°{to_u.upper()[0]}"
