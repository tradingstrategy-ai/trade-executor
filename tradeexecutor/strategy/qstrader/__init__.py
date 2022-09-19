"""QSTrader portfolio construction model based strategy types."""


try:
    import qstrader
    #: A flag to tell if we are installed with the optional qstrader dependencies
    #: This is optional, because QSTrader pulls in a lot of stuff we don't
    #: really care about
    HAS_QSTRADER = True
except ImportError:
    HAS_QSTRADER = False

