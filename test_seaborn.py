try:
    import seaborn as sns
    print('SUCCESS: Seaborn version:', sns.__version__)
except Exception as e:
    print('ERROR:', e)
