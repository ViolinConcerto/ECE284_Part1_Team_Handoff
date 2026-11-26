module mac #(
    parameter bw = 4,
    parameter psum_bw = 32
)(
    output signed [psum_bw-1:0] out,
    input [bw-1:0] a,
    input signed [bw-1:0] b,
    input signed [psum_bw-1:0] c,
    input act_signed
);
    wire signed [bw:0] activation;
    wire signed [2*bw:0] product;
    wire signed [psum_bw-1:0] product_extended;
    assign activation = act_signed ? {a[bw-1], a} : {1'b0, a};
    assign product = activation * b;
    assign product_extended = product;
    assign out = product_extended + c;
endmodule
