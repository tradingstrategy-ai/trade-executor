"""Router functions.

- Interface for the route chooser function :py:class:`RoutingFunction`

- The default router choose :py:func:`default_route_chooser`
"""


class UnroutableTrade(Exception):
    """Trade cannot be routed, as we could not find a matching route.

    """




