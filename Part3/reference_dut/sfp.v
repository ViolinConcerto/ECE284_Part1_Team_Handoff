// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module sfp (out, in, acc, clear, relu, clk, reset);

parameter bw = 8;
parameter psum_bw = 16;

input clk;
input acc;
input clear;
input relu;
input reset;

input signed [psum_bw-1:0] in;
output  signed [psum_bw-1:0] out;
reg  signed [psum_bw-1:0] psum_q;

always @(posedge clk) begin
    if (reset || clear) begin
        psum_q <= 0;
    end else if (acc) begin
        psum_q <= psum_q + in;
    end else if (relu) begin
        psum_q <= (psum_q > 0) ? psum_q : 0;
    end
end

assign out = psum_q;

endmodule
