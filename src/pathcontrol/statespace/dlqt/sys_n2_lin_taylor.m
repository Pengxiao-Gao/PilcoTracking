clear all
close all
clc


%%

syms x_k y_k theta_k Ts v lo u_k


f_k = [x_k + Ts*(v*cos(theta_k) - lo*sin(theta_k)*u_k );
       y_k + Ts*(v*sin(theta_k) + lo*cos(theta_k)*u_k );
       theta_k + Ts*u_k]
   
A = jacobian(f_k, [x_k, y_k, theta_k])
B = jacobian(f_k, [u_k])

















