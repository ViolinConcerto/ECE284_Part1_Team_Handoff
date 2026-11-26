// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
module ofifo (clk, in, out, rd, wr, o_full, reset, o_ready, o_valid);
  parameter col = 8;
  parameter bw = 4;

  input clk, rd, reset;
  input [col-1:0] wr;
  input [col*bw-1:0] in;
  output [col*bw-1:0] out;
  output o_full, o_ready, o_valid;

  wire [col-1:0] full, empty;
  reg rd_en;

  assign o_full = |full;
  assign o_ready = ~|full;
  assign o_valid = ~|empty;

  genvar i;
  for (i = 0; i < col; i = i + 1) begin : col_num
    fifo_depth64 #(.bw(bw)) fifo_instance (
      .rd_clk(clk), .wr_clk(clk), .reset(reset),
      .wr(wr[i]), .rd(rd_en),
      .in(in[i*bw +: bw]), .out(out[i*bw +: bw]),
      .o_full(full[i]), .o_empty(empty[i])
    );
  end

  always @(posedge clk) begin
    if (reset)
      rd_en <= 1'b0;
    else
      rd_en <= rd;
  end
endmodule
