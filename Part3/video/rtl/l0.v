module l0 #(
    parameter row = 8,
    parameter bw = 4
)(
    input clk, reset, rd, wr,
    input [row*bw-1:0] in,
    output [row*bw-1:0] out,
    output o_full, o_ready
);
    wire [row-1:0] empty, full;
    reg [row-1:0] rd_en;
    assign o_full = |full;
    assign o_ready = ~(|full);
    genvar r;
    generate for (r=0; r<row; r=r+1) begin : row_num
        fifo_depth64 #(.bw(bw)) fifo_instance(
            .rd_clk(clk), .wr_clk(clk), .rd(rd_en[r]), .wr(wr),
            .o_empty(empty[r]), .o_full(full[r]),
            .in(in[r*bw +: bw]), .out(out[r*bw +: bw]), .reset(reset));
    end endgenerate
    integer r_index;
    always @(posedge clk) begin
        if (reset) rd_en <= 0;
        else begin
            rd_en[0] <= rd;
            for (r_index=1; r_index<row; r_index=r_index+1)
                rd_en[r_index] <= rd_en[r_index-1];
        end
    end
endmodule
