from argparse import ArgumentParser


def get_args():
    """Getter of args

    Returns:
        argparse.Namespace: from console
    """
    argparser = ArgumentParser()
    argparser.add_argument('version',
                           type=str,
                           help='Version ID')
    return argparser.parse_args()


def get_version():
    """Getter of an input version

    Returns:
        str: version of an input file
    """
    return get_args().version
