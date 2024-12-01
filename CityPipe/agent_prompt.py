reflect_system_prompt = '''
You are an emergency response agent in a city environment. You need to analyze the action history compared with the task description and milestone description to evaluate whether the emergency response task is completed.
The check-structure
{
    "reasoning": str, # detailed analysis of the response actions taken
    "summary": str, # summary of key actions taken, including locations, resources deployed, and response times
    "task_status": bool, # whether the emergency response task is completed
}
'''

reflect_user_prompt = '''
Now you have attempted to handle the emergency situation. 
The task description is:
{{task_description}}

The milestone description is:
{{milestone_description}}

The action history is:
{{state}}
{{action_history}}

Please evaluate whether the emergency response task is completed and return a check-structure json.
'''

city_emergency_knowledge = '''
Here are some key points about urban emergency response:
1. Emergency Types: Fire, medical emergency, traffic accident, hazardous material spill, etc.
2. Response Units: Ambulances, fire trucks, police cars, and specialized response teams.
3. Resource Management: Each unit has limited capacity and requires coordination for optimal deployment.
4. Time Sensitivity: Response time is critical - faster response typically means better outcomes.
5. Coordination: Multiple agencies need to work together (medical, fire, police, traffic control).
6. Safety Zones: Establish and maintain safety perimeters around incident sites.
7. Communication: Clear communication channels between all responding units is essential.
8. Resource Prioritization: Critical cases take precedence, but maintain reserve capacity for new emergencies.
'''


agent_prompt = ''' 
*** The relevant emergency data ***
{{relevant_data}}

*** Coordinating response teams ***
{{other_agents}}

*** {{agent_name}}'s current status ***
{{agent_state}}

*** Recent response actions taken ***
{{agent_action_list}}

*** Emergency environment status ***
{{env}}

*** Emergency response knowledge ***
{{city_emergency_knowledge}}

*** Emergency task description *** 
{{task_description}}


Guidelines for response:
1. Assess situation severity and required resources
2. Coordinate with other response units
3. Prioritize life-saving actions
4. Maintain clear communication
5. Follow standard emergency protocols
6. Document all actions taken

Respond with clear, actionable steps to address the emergency situation.
'''

agent_cooper_prompt = ''' 
*** The relevant emergency data ***
{{relevant_data}}

*** Coordinating response teams ***
{{other_agents}}

*** {{agent_name}}'s current status ***
{{agent_state}}

*** Recent response actions taken ***
{{agent_action_list}}

*** Emergency environment status ***
{{env}}

*** Emergency response knowledge ***
{{city_emergency_knowledge}}

*** Emergency task description *** 
{{task_description}}

You need to work as the leader use api control your team(include yourself and other agents) to complete the task.
Your team members are:
{{team_members}}
At least two Action before the Final Answer.
'''