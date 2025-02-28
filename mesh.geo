lc = 1.0;  // Smaller value = finer mesh

Point(1) = {0, 2, 0, lc};
Point(2) = {10, 2, 0, lc};
Point(3) = {10, 0, 0, lc};
Point(4) = {1, 0, 0, lc};
Point(5) = {1, 1, 0, lc};
Point(6) = {-0, 1, 0, lc};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line Loop(7) = {1, 2, 3, 4, 5, 6};
Plane Surface(8) = {7};
Physical Line("Inflow") = {6};
Physical Line("NoSlip") = {1, 3, 4, 5};
Physical Line("Outflow") = {2};
Physical Surface("Channel") = {8};

