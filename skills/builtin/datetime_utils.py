"""Date and time utilities skill."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from skills import skill


@skill(
    name="datetime",
    description="Get current time, convert timezones, calculate date differences",
    version="1.0.0",
    author="MAUDE",
    triggers=["time", "date", "timezone", "when"],
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["now", "convert", "diff", "add"],
                "description": "Action to perform",
                "default": "now"
            },
            "timezone": {
                "type": "string",
                "description": "Timezone (e.g., 'America/New_York', 'Europe/London', 'UTC')"
            },
            "from_tz": {
                "type": "string",
                "description": "Source timezone for conversion"
            },
            "to_tz": {
                "type": "string",
                "description": "Target timezone for conversion"
            },
            "date1": {
                "type": "string",
                "description": "First date (YYYY-MM-DD format)"
            },
            "date2": {
                "type": "string",
                "description": "Second date (YYYY-MM-DD format)"
            },
            "days": {
                "type": "integer",
                "description": "Number of days to add/subtract"
            }
        }
    }
)
def datetime_utils(
    action: str = "now",
    timezone: str = None,
    from_tz: str = None,
    to_tz: str = None,
    date1: str = None,
    date2: str = None,
    days: int = None
) -> str:
    """Date/time utilities."""

    if action == "now":
        return _get_current_time(timezone)
    elif action == "convert":
        return _convert_timezone(from_tz, to_tz)
    elif action == "diff":
        return _date_difference(date1, date2)
    elif action == "add":
        return _add_days(date1, days)
    else:
        return f"Unknown action: {action}"


def _get_current_time(timezone: str = None) -> str:
    """Get current time in specified timezone."""
    try:
        if timezone:
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
            return f"Current time in {timezone}:\n  {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        else:
            # Show multiple common timezones
            zones = [
                ("Local", None),
                ("UTC", "UTC"),
                ("New York", "America/New_York"),
                ("London", "Europe/London"),
                ("Tokyo", "Asia/Tokyo"),
            ]
            lines = ["Current time:"]
            for name, tz_name in zones:
                if tz_name:
                    tz = ZoneInfo(tz_name)
                    now = datetime.now(tz)
                else:
                    now = datetime.now()
                lines.append(f"  {name:12}: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}. Use timezone like 'America/New_York', 'Europe/London', 'UTC'"


def _convert_timezone(from_tz: str, to_tz: str) -> str:
    """Convert current time between timezones."""
    if not from_tz or not to_tz:
        return "Error: Please specify from_tz and to_tz parameters"

    try:
        from_zone = ZoneInfo(from_tz)
        to_zone = ZoneInfo(to_tz)

        now_from = datetime.now(from_zone)
        now_to = now_from.astimezone(to_zone)

        return (
            f"Time conversion:\n"
            f"  {from_tz}: {now_from.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"  {to_tz}: {now_to.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
    except Exception as e:
        return f"Error: {e}"


def _date_difference(date1: str, date2: str = None) -> str:
    """Calculate difference between two dates."""
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d") if date1 else datetime.now()
        d2 = datetime.strptime(date2, "%Y-%m-%d") if date2 else datetime.now()

        diff = d2 - d1
        days = abs(diff.days)

        weeks = days // 7
        remaining_days = days % 7

        direction = "from now" if diff.days > 0 else "ago"
        if date2:
            direction = "between dates"

        return (
            f"Date difference:\n"
            f"  From: {d1.strftime('%Y-%m-%d')}\n"
            f"  To: {d2.strftime('%Y-%m-%d')}\n"
            f"  Difference: {days} days ({weeks} weeks, {remaining_days} days)"
        )
    except ValueError:
        return "Error: Use YYYY-MM-DD date format"
    except Exception as e:
        return f"Error: {e}"


def _add_days(date1: str = None, days: int = None) -> str:
    """Add days to a date."""
    if days is None:
        return "Error: Please specify 'days' parameter"

    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d") if date1 else datetime.now()
        result = d1 + timedelta(days=days)

        action = "Adding" if days >= 0 else "Subtracting"
        return (
            f"{action} {abs(days)} days:\n"
            f"  From: {d1.strftime('%Y-%m-%d (%A)')}\n"
            f"  Result: {result.strftime('%Y-%m-%d (%A)')}"
        )
    except ValueError:
        return "Error: Use YYYY-MM-DD date format"
    except Exception as e:
        return f"Error: {e}"
