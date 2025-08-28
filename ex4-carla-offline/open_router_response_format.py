class ResponseFormat():
    planner = {
        "type": "json_schema",
        'json_schema': {
            'name'  : 'planner',
            'strict': True,
            'schema': {
                'type'      : 'object',
                'properties': {
                    'task':{
                        'type': 'string',
                        'description': 'The task to be performed'
                    },
                    'steps': {
                        'type': 'array',
                        'items': {
                            'type': 'string',
                            'description': 'A step in the task'
                        }
                    }
                },
                'required': ['task', 'steps']
            }
        }
    }
    researcher = {
        "type": "json_schema",
        'json_schema': {
            'name'  : 'researcher',
            'strict': True,
            'schema': {
                'type'      : 'object',
                'properties': {
                    'findings': {
                        'type': 'array',
                        'items': {
                            'type': 'string',
                            'description': 'A finding in the research'
                        }
                    }
                },
                'required': ['findings']
            }
        }
    }
    critic = {
        "type": "json_schema",
        'json_schema': {
            'name'  : 'critic',
            'strict': True,
            'schema': {
                'type'      : 'object',
                'properties': {
                    'critique': {
                        'type': 'string',
                        'description': 'A refinement of the research'
                    }
                },
                'required': ['critique']
            }
        }
    }

