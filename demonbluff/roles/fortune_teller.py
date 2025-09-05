from . import Role, register_role

@register_role
class FortuneTeller(Role):
    name = "fortune teller"

    @staticmethod
    def static_constraints(world, puzzle):
        return []

    def simulate_action(self, world, action_args):
        return []
