class LLMPlan():
    def __init__(self, llm_return):
        # parse llm_return to format, like:
        # llm_return = """
        # A) varian_1
        # B) varian_2
        # C) variant_3
        # D) variant_4
        # """
        #llm_return_list = ['varian_1', 'variant_2', 'variant_3', 'variant_4']
        # instead A, B, C, D - may be a,b,c,d,1,2,3,4 and so on
    pass