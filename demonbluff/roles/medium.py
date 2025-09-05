from . import Role, register_role

@register_role
class Medium(Role):
    name = "medium"

    @staticmethod
    def static_constraints(world, puzzle):
        return []

    def simulate_action(self, world, action_args):
        return []
