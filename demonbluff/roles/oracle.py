from . import Role, register_role

@register_role
class Oracle(Role):
    name = "oracle"

    @staticmethod
    def static_constraints(world, puzzle):
        return []

    def simulate_action(self, world, action_args):
        return []
