for (int i = 1; i < local_x_dim-1; i++)
{
    A_descent.set(i, 0, -(descent.get(i+1, 0) + descent.get(i-1, 0)) * inv_h_x_sqr - (descent.get(i, 1) + neighbors_descent.get(1, i)) * inv_h_y_sqr + D.get(i + x_range_min, y_range_min)*descent.get(i, 0));
    A_descent.set(i, local_y_dim-1, -(descent.get(i+1, local_y_dim-1) + descent.get(i-1, local_y_dim-1)) * inv_h_x_sqr - (neighbors_descent.get(0, i) + descent.get(i, local_y_dim-1-1)) * inv_h_y_sqr + D.get(i + x_range_min, local_y_dim-1 + y_range_min)*descent.get(i, local_y_dim-1));
    descent_A_descent += descent.get(i, 0) * A_descent.get(i, 0);
    descent_A_descent += descent.get(i, local_y_dim-1) * A_descent.get(i, local_y_dim-1);
}

for (int j = 1; j < local_y_dim-1; j++)
{
    A_descent.set(0, j, -(descent.get(1, j) + neighbors_descent.get(2, j)) * inv_h_x_sqr - (descent.get(0, j+1) + descent.get(0, j-1)) * inv_h_y_sqr + D.get(x_range_min, j + y_range_min)*descent.get(0, j));
    A_descent.set(local_x_dim-1, j, -(neighbors_descent.get(3, j) + descent.get(local_x_dim-1-1, j)) * inv_h_x_sqr - (descent.get(local_x_dim-1, j+1) + descent.get(local_x_dim-1, j-1)) * inv_h_y_sqr + D.get(local_x_dim-1 + x_range_min, j + y_range_min)*descent.get(local_x_dim-1, j));
    descent_A_descent += descent.get(0, j) * A_descent.get(0, j);
    descent_A_descent += descent.get(local_x_dim-1, j) * A_descent.get(local_x_dim-1, j);
}

A_descent.set(0, 0, -(descent.get(1, 0) + neighbors_descent.get(2, 0)) * inv_h_x_sqr - (descent.get(0, 1) + neighbors_descent.get(1, 0)) * inv_h_y_sqr + D.get(x_range_min, y_range_min)*descent.get(0, 0));
A_descent.set(local_x_dim-1, 0, -(neighbors_descent.get(3, 0) + descent.get(local_x_dim-1-1, 0)) * inv_h_x_sqr - (descent.get(local_x_dim-1, 1) + neighbors_descent.get(1, local_x_dim-1)) * inv_h_y_sqr + D.get(local_x_dim-1 + x_range_min, y_range_min)*descent.get(local_x_dim-1, 0));
A_descent.set(local_x_dim-1, local_y_dim-1, -(neighbors_descent.get(3, local_y_dim-1) + descent.get(local_x_dim-1-1, local_y_dim-1)) * inv_h_x_sqr - (neighbors_descent.get(0, local_x_dim-1) + descent.get(local_x_dim-1, local_y_dim-1-1)) * inv_h_y_sqr + D.get(local_x_dim-1 + x_range_min, local_y_dim-1 + y_range_min)*descent.get(local_x_dim-1, local_y_dim-1));
A_descent.set(0, local_y_dim-1, -(descent.get(1, local_y_dim-1) + neighbors_descent.get(2, local_y_dim-1)) * inv_h_x_sqr - (neighbors_descent.get(0, 0) + descent.get(0, local_y_dim-1-1)) * inv_h_y_sqr + D.get(x_range_min, local_y_dim-1 + y_range_min)*descent.get(0, local_y_dim-1));    

descent_A_descent += descent.get(0, 0) * A_descent.get(0, 0);
descent_A_descent += descent.get(local_x_dim-1, 0) * A_descent.get(local_x_dim-1, 0);
descent_A_descent += descent.get(local_x_dim-1, local_y_dim-1) * A_descent.get(local_x_dim-1, local_y_dim-1);
descent_A_descent += descent.get(0, local_y_dim-1) * A_descent.get(0, local_y_dim-1);