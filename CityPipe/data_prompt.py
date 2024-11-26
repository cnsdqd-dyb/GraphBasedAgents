AGENT_INFO_STORE_FORMAT = """{name} is currently at {position}. {name} has access to {resources}. Current status: {status}. Available emergency units: {units}. {nearby_info}
The emergency situation in the vicinity:
{emergency_info}
"""

SOMEONE_NEARBY_INFO_FORMAT = "Other response units in the area: {unit_list}."

NOONE_NEARBY_INFO_FORMAT = "No other response units in the immediate vicinity."

SUCCESS_DECOMPOSE_PLAN_FORMAT = """Here is a decompose plan:
Nodes: {nodes}
Edges: {edges}
Entry: {entry}
Exit: {exit}

The response plan has been successfully implemented.
"""

NOT_SUCCESS_DECOMPOSE_PLAN_FORMAT = """Here is a decompose plan:
Nodes: {nodes}
Edges: {edges}
Entry: {entry}
Exit: {exit}

The response plan is currently in progress.
"""

RESPONDER_INFO_FORMAT = """{name} is at location {position} with {resources} available"""

EMERGENCY_INFO_FORMAT = """At {time}, {responder_info}. Current emergency status: {emergency_status}. Affected areas: {affected_areas}. Casualties: {casualties}. Risk level: {risk_level}."""

SUMMARY_ENVIRONMENT_SYSTEM_PROMPT = """You are an emergency response coordinator.
Based on the emergency situation and response requirements, extract key information and provide a concise yet comprehensive summary of the current environment.
Focus on critical factors such as casualties, hazards, available resources, and response unit positions.
"""

SUMMARY_ENVIRONMENT_EXAMPLE_PROMPT = [
    """The emergency info:
{"responder_info": [{"name": "MedicalTeam1", "position": [42.5, 73.2], "resources": {"ambulances": 2, "paramedics": 4}}], 
"emergency_status": {"type": "fire", "severity": "high", "location": [41.8, 72.9]}, 
"affected_areas": ["residential_building_A", "commercial_zone_B"], 
"time": "14:30",
"casualties": {"confirmed": 2, "potential": 5},
"risk_level": "high",
"nearby_units": [{"FireTeam1": [42.1, 73.0]}, {"PoliceUnit3": [42.3, 73.1]}]}
*** The task ***: Coordinate emergency response for a high-rise building fire.
""",
    """The summary of the emergency situation:
Response Units: MedicalTeam1 at [42.5, 73.2] with 2 ambulances and 4 paramedics, FireTeam1 and PoliceUnit3 nearby.
Emergency: High-severity fire at [41.8, 72.9] affecting residential and commercial zones.
Casualties: 2 confirmed, 5 potential victims.
Risk Level: High, requiring immediate multi-unit response.
Time Critical: Incident reported at 14:30, immediate action required.
"""]

ENVIRONMENT_UPDATE_FORMAT = """
Current Time: {time}
Location: {location}
Emergency Status:
- Type: {emergency_type}
- Severity: {severity}
- Affected Area: {affected_area}
- Casualties: {casualties}

Available Resources:
{resources}

Response Units:
{units}

Critical Updates:
{updates}
"""

HISTORY_SUMMARY_PROMPT = """You are {name}. Your task is to create a concise running summary of actions and information results in the provided text, focusing on key and potentially important information to remember.

You will receive the current summary and the your latest actions. Combine them, adding relevant key information from the latest development in 1st person past tense and keeping the summary concise.
The subject of the sentence should be {name}.

Summary So Far:
{summary_so_far}

Latest Development:
{latest_development}

Your Summary:
"""
