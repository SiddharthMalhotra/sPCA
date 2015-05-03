%compare ppca vs. svd using water.data

%Y=rand(527,8);
%Y=[Y, Y.*2, Y.*3, Y.*4, Y.*5];
 Y=load("water.data");
 %Y=rand(5);
 d=8
 pr=min(10,size(Y));
 Y(isnan(Y))=0;
 my=mean(Y);
 dy=max(Y)-min(Y);
 ny=(Y-my)./dy;
 [C, ss, M, X, Ye] = newppca(ny, d, 1); X(1:pr,:)
 norm(C(:,1))
 [U S V] = svd(ny');
 norm(U(:,1))
 t =ny * U(:,1:d); t(1:pr,:)
 diff_svd_ppc           = norm(X-t,1)
 diff_abs_svd_ppc       = norm(abs(X)-abs(t),1)
 norm_ny                = norm(ny, 1)
 construction_error_ppc = norm(ny - X * C', 1)
 construction_error_svd = norm(ny - t * U(:,1:d)', 1)

