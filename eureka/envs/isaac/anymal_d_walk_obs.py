class AnymalDWalk(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        self.measured_heights = self.get_heights()
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.height_meas_scale
        self.obs_buf = torch.cat((  self.base_lin_vel * self.lin_vel_scale,
                                    self.base_ang_vel  * self.ang_vel_scale,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos )* self.dof_pos_scale,
                                    self.dof_vel * self.dof_vel_scale,
                                    self.actions,
                                    heights
                                    ), dim=-1)
