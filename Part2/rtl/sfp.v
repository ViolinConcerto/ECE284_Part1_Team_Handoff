module sfp #(
    parameter bw = 4,
    parameter psum_bw = 32
)(
    input clk, reset, clear, acc, relu,
    input [4:0] shift,
    input signed [psum_bw-1:0] in,
    output signed [psum_bw-1:0] out
);
    reg signed [psum_bw-1:0] psum_q;
    wire signed [psum_bw-1:0] wide_input = in;
    wire signed [psum_bw-1:0] shifted_input = wide_input <<< shift;
    always @(posedge clk) begin
        if (reset || clear) psum_q <= 0;
        else if (acc) psum_q <= psum_q + shifted_input;
        else if (relu) psum_q <= (psum_q > 0) ? psum_q : 0;
    end
    assign out = psum_q;
endmodule
