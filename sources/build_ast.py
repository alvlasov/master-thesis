import east

def build_ast(data):
    i, text = data
    return i, east.asts.base.AST.get_ast(east.utils.text_to_strings_collection(text, words=5))
