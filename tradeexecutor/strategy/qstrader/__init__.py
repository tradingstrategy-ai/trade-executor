"""QSTrader portfolio construction model based strategy types."""


try:
    import qstrader
    HAS_QSTRADER = True
except ImportError:
    HAS_QSTRADER = False

