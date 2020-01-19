function x = argmin(U,dU, x0)
  while 1
    x = x0 - 0.01*dU(x0);
    if abs(U(x0)-U(x))< 1e-9
      break;
    end
    x0 = x;
  end
end
