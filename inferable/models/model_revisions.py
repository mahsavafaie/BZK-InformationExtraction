
MODEL_REVISIONS = {
    # InternVL 2
    'OpenGVLab/InternVL2-1B': '3bc2e85',
    'OpenGVLab/InternVL2-2B': 'aec61df',
    'OpenGVLab/InternVL2-4B': 'e5a72e2',
    'OpenGVLab/InternVL2-8B': '939880e',
    'OpenGVLab/InternVL2-26B': 'e3247d2',
    'OpenGVLab/InternVL2-40B': '87411dc',
    'OpenGVLab/InternVL2-Llama3-76B': '02e7994',

    # InternVL 2.5
    'OpenGVLab/InternVL2_5-1B': '4dcf984',
    'OpenGVLab/InternVL2_5-2B': 'fc8d7d6',
    'OpenGVLab/InternVL2_5-4B': 'd8e5979',
    'OpenGVLab/InternVL2_5-8B': 'd64b85a',
    'OpenGVLab/InternVL2_5-26B': '73bb495',
    'OpenGVLab/InternVL2_5-38B': '07c2078',
    'OpenGVLab/InternVL2_5-78B': 'ea891f5'
}


def get_model_revision(model_name):
    return MODEL_REVISIONS.get(model_name, "main")