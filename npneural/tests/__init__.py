# Importing the module will execute 
# outer level statements

def test_mnist():
    from .test_mnist     import run_test
    run_test()


def test_spiral():
    from .test_spiral    import run_test
    run_test()


def test_vertical():
    from .test_vertical  import run_test
    run_test()


def test_xor():
    from .test_xor       import run_test
    run_test()


